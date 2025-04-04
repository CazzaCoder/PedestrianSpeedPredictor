import SwiftUI
import RealityKit
import ARKit
import Vision
import CoreML

// MARK: - ContentView
struct ContentView: View {
    var body: some View {
        ARViewContainer()
            .edgesIgnoringSafeArea(.all)
    }
}

// MARK: - ARViewContainer
struct ARViewContainer: UIViewRepresentable {
    func makeUIView(context: Context) -> CustomARView {
        CustomARView(frame: .zero)
    }
    
    func updateUIView(_ uiView: CustomARView, context: Context) {}
}

// MARK: - PersonTrack
class PersonTrack {
    let id: UUID
    var lastObservation: VNDetectedObjectObservation
    var position: SIMD3<Float>?
    var lastUpdateTime: TimeInterval
    var anchor: AnchorEntity?
    var entity: ModelEntity?
    var speed: Float = 0.0
    var direction: SIMD3<Float> = .zero
    var lastConfidence: Float = 0.0
    var arrowEntity: ModelEntity?
    
    init(id: UUID, observation: VNDetectedObjectObservation, time: TimeInterval) {
        self.id = id
        self.lastObservation = observation
        self.lastUpdateTime = time
    }
    
    func removeFromScene(_ scene: RealityKit.Scene) {
        anchor?.removeFromParent()
    }
}

// MARK: - CustomARView
class CustomARView: ARView {
    private var visionRequests: [VNRequest] = []
    private let model: VNCoreMLModel
    private var tracks: [PersonTrack] = []
    private var frameCounter = 0
    private let detectionInterval = 5
    private var lastFrameTimestamp: CFTimeInterval = 0.0
    
    required init(frame frameRect: CGRect) {
        do {
            model = try VNCoreMLModel(for: yolo11n_640_nms(configuration: .init()).model)
        } catch {
            fatalError("Failed to load YOLO model: \(error)")
        }
        super.init(frame: frameRect)
        
        let config = ARWorldTrackingConfiguration()
        config.planeDetection = [.horizontal]
        session.run(config)
        
        setupVision()
        session.delegate = self
    }
    
    @objc required dynamic init?(coder decoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    // MARK: Vision Setup
    private func setupVision() {
        let request = VNCoreMLRequest(model: model) { [weak self] request, error in
            guard let self = self else { return }
            if let error = error {
                print("Vision request failed: \(error)")
                return
            }
            guard let results = request.results else {
                print("No results from Vision request")
                return
            }
            
            // Print the overall type and count of results
            print("Results type: \(type(of: results))")
            print("Number of results: \(results.count)")
            
            // Optional: Proceed with person filtering if detections are present
            if let observations = results as? [VNRecognizedObjectObservation] {
                let personObservations = observations.filter {
                    $0.labels.first?.identifier.lowercased() == "person" && $0.confidence > 0.5
                }
                print("Filtered person observations: \(personObservations.count)")
                self.handleNewDetections(personObservations)
            } else {
                print("No VNRecognizedObjectObservation found in results")
            }
        }
        request.imageCropAndScaleOption = .scaleFill
        visionRequests = [request]
    }
    
    // MARK: Detection Handling
    private func handleNewDetections(_ detections: [VNRecognizedObjectObservation]) {
        print("[handleNewDetections] Received \(detections.count) detections")
        var newTracks: [PersonTrack] = []
        let currentTime = CACurrentMediaTime()
        
        for detection in detections {
            var matched = false
            for track in tracks {
                let iou = computeIoU(box1: detection.boundingBox, box2: track.lastObservation.boundingBox)
                if iou > 0.5 {
                    track.lastObservation = detection
                    track.lastConfidence = detection.confidence
                    track.lastUpdateTime = currentTime
                    matched = true
                    break
                }
            }
            if !matched {
                let newTrack = PersonTrack(id: UUID(), observation: detection, time: currentTime)
                newTrack.lastConfidence = detection.confidence
                newTracks.append(newTrack)
            }
        }
        tracks.append(contentsOf: newTracks)
    }
    
    private func computeIoU(box1: CGRect, box2: CGRect) -> Float {
        let intersection = box1.intersection(box2)
        guard !intersection.isNull else { return 0.0 }
        let intersectionArea = intersection.width * intersection.height
        let unionArea = (box1.width * box1.height) + (box2.width * box2.height) - intersectionArea
        return Float(intersectionArea / unionArea)
    }
    
    // MARK: Track Updates
    private func updateTrackPositionAndSpeed(track: inout PersonTrack, frame: ARFrame) {
        let boundingBox = track.lastObservation.boundingBox
        // Adjust to slightly above the bottom for better foot positioning
        let centerVision = CGPoint(x: boundingBox.midX, y: boundingBox.minY + boundingBox.height * 0.1)
        let viewSize = self.frame.size
        let centerPixel = CGPoint(x: centerVision.x * viewSize.width, y: (1 - centerVision.y) * viewSize.height)
        
        guard let rayResult = ray(through: centerPixel) else { return }
        
        let query = ARRaycastQuery(origin: rayResult.origin, direction: rayResult.direction, allowing: .estimatedPlane, alignment: .horizontal)
        guard let result = session.raycast(query).first else { return }
        
        let newPosition = SIMD3(result.worldTransform.columns.3.x, result.worldTransform.columns.3.y, result.worldTransform.columns.3.z)
        
        if let oldPosition = track.position, frame.timestamp > track.lastUpdateTime {
            let dt = Float(frame.timestamp - track.lastUpdateTime)
            if dt > 0 {
                let distance = simd_length(newPosition - oldPosition)
                track.speed = distance / dt
                let rawDirection = newPosition - oldPosition
                let alpha: Float = 0.7
                track.direction = alpha * track.direction + (1 - alpha) * (dt > 0 ? simd_normalize(rawDirection) : .zero)
            }
        }
        track.position = newPosition
        track.lastUpdateTime = frame.timestamp
        updateEntityForTrack(&track)
    }
    
    private func updateEntityForTrack(_ track: inout PersonTrack) {
        let speedString = String(format: "%.2f m/s", track.speed)
        
        if let anchor = track.anchor, let entity = track.entity, let position = track.position {
            anchor.position = position
            entity.position = SIMD3<Float>(0, 1.0, 0) // Lowered from 1.5 for better visibility
            
            let newMesh = MeshResource.generateText(
                speedString,
                font: .systemFont(ofSize: 0.05),
                containerFrame: .zero,
                alignment: .center,
                lineBreakMode: .byWordWrapping
            )
            entity.model?.mesh = newMesh
            
            let arrowLength = track.speed * 0.5
            let arrowDirection = track.direction
            
            if track.arrowEntity == nil {
                let shaftRadius: Float = 0.01
                let shaftHeight: Float = 0.5
                let coneHeight: Float = 0.1
                let shaftMesh = MeshResource.generateCylinder(height: shaftHeight, radius: shaftRadius)
                let coneMesh = MeshResource.generateCone(height: coneHeight, radius: 0.025)
                let shaftEntity = ModelEntity(mesh: shaftMesh, materials: [SimpleMaterial(color: .blue, isMetallic: false)])
                shaftEntity.position.y = shaftHeight / 2
                let coneEntity = ModelEntity(mesh: coneMesh, materials: [SimpleMaterial(color: .blue, isMetallic: false)])
                coneEntity.position.y = shaftHeight
                let arrowParent = ModelEntity()
                arrowParent.addChild(shaftEntity)
                arrowParent.addChild(coneEntity)
                arrowParent.scale = SIMD3<Float>(repeating: 0.001)
                anchor.addChild(arrowParent)
                track.arrowEntity = arrowParent
            }
            
            if let arrowEntity = track.arrowEntity {
                let arrowOpacity: Float = track.lastConfidence < 0.5 ? 0.3 : 1.0
                let materials = [SimpleMaterial(color: .blue.withAlphaComponent(CGFloat(arrowOpacity)), isMetallic: false)]
                for child in arrowEntity.children {
                    if var model = child as? ModelEntity {
                        model.model?.materials = materials
                    }
                }
                let baseHeight: Float = 0.6
                let targetScale = arrowLength / baseHeight
                arrowEntity.scale = SIMD3<Float>(repeating: 0.001 + max(targetScale, 0.0))
                
                if simd_length(arrowDirection) > 0.0001 {
                    let up = SIMD3<Float>(0, 1, 0)
                    let dirNorm = simd_normalize(arrowDirection)
                    let dotVal = simd_dot(up, dirNorm)
                    let angle = acos(dotVal)
                    let axis = simd_normalize(simd_cross(up, dirNorm))
                    if !axis.allFinite {
                        arrowEntity.orientation = simd_quatf(angle: angle, axis: axis)
                    }
                }
                arrowEntity.position = .zero
            }
        } else if let position = track.position {
            let anchor = AnchorEntity(world: position)
            let labelEntity = ModelEntity(
                mesh: MeshResource.generateText(
                    speedString,
                    font: .systemFont(ofSize: 0.05),
                    containerFrame: .zero,
                    alignment: .center,
                    lineBreakMode: .byWordWrapping
                ),
                materials: [SimpleMaterial(color: .white, isMetallic: false)]
            )
            labelEntity.scale = SIMD3<Float>(repeating: 0.1)
            labelEntity.position = SIMD3<Float>(0, 1.0, 0)
            anchor.addChild(labelEntity)
            scene.addAnchor(anchor)
            track.anchor = anchor
            track.entity = labelEntity
            
            let arrowLength = track.speed * 0.5
            let arrowDirection = track.direction
            
            let shaftRadius: Float = 0.01
            let shaftHeight: Float = 0.5
            let coneHeight: Float = 0.1
            let shaftMesh = MeshResource.generateCylinder(height: shaftHeight, radius: shaftRadius)
            let coneMesh = MeshResource.generateCone(height: coneHeight, radius: 0.025)
            let shaftEntity = ModelEntity(mesh: shaftMesh, materials: [SimpleMaterial(color: .blue, isMetallic: false)])
            shaftEntity.position.y = shaftHeight / 2
            let coneEntity = ModelEntity(mesh: coneMesh, materials: [SimpleMaterial(color: .blue, isMetallic: false)])
            coneEntity.position.y = shaftHeight
            let arrowParent = ModelEntity()
            arrowParent.addChild(shaftEntity)
            arrowParent.addChild(coneEntity)
            arrowParent.scale = SIMD3<Float>(repeating: 0.001)
            anchor.addChild(arrowParent)
            track.arrowEntity = arrowParent
            
            if let arrowEntity = track.arrowEntity {
                let arrowOpacity: Float = track.lastConfidence < 0.5 ? 0.3 : 1.0
                let materials = [SimpleMaterial(color: .blue.withAlphaComponent(CGFloat(arrowOpacity)), isMetallic: false)]
                for child in arrowEntity.children {
                    if var model = child as? ModelEntity {
                        model.model?.materials = materials
                    }
                }
                let baseHeight: Float = 0.6
                let targetScale = arrowLength / baseHeight
                arrowEntity.scale = SIMD3<Float>(repeating: 0.001 + max(targetScale, 0.0))
                
                if simd_length(arrowDirection) > 0.0001 {
                    let up = SIMD3<Float>(0, 1, 0)
                    let dirNorm = simd_normalize(arrowDirection)
                    let dotVal = simd_dot(up, dirNorm)
                    let angle = acos(dotVal)
                    let axis = simd_normalize(simd_cross(up, dirNorm))
                    if !axis.allFinite {
                        arrowEntity.orientation = simd_quatf(angle: angle, axis: axis)
                    }
                }
                arrowEntity.position = .zero
            }
        }
    }
}

// MARK: - ARSessionDelegate
extension CustomARView: ARSessionDelegate {
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let deltaT = frame.timestamp - lastFrameTimestamp
//        print("[INFO] Frame Time Delta: \(String(format: "%.3f", deltaT)) seconds")
        lastFrameTimestamp = frame.timestamp
        
        autoreleasepool {
            frameCounter += 1
            let pixelBuffer = frame.capturedImage
            let orientation = UIDevice.current.orientation.cgImageOrientation
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: orientation, options: [:])
            
            var requests: [VNRequest] = []
            if frameCounter % detectionInterval == 0 {
                requests.append(visionRequests[0])
            }
            
            for track in tracks {
                let trackingRequest = VNTrackObjectRequest(detectedObjectObservation: track.lastObservation)
                trackingRequest.trackingLevel = .accurate
                requests.append(trackingRequest)
            }
            
            do {
                try handler.perform(requests)
//                print("[INFO] Frame #\(frameCounter) processed at timestamp \(frame.timestamp)")
                
                for request in requests where request is VNTrackObjectRequest {
                    if let result = request.results?.first as? VNDetectedObjectObservation,
                       let index = tracks.firstIndex(where: { $0.lastObservation == (request as! VNTrackObjectRequest).inputObservation }) {
                        tracks[index].lastObservation = result
                        updateTrackPositionAndSpeed(track: &tracks[index], frame: frame)
                    }
                }
                
                let currentTime = frame.timestamp
                let activeTracks = tracks.filter { currentTime - $0.lastUpdateTime < 1.0 }
                for track in tracks where !activeTracks.contains(where: { $0.id == track.id }) {
                    track.removeFromScene(scene)
                }
                tracks = activeTracks
                
            } catch {
                print("Failed to perform Vision requests: \(error)")
            }
        }
    }
}

// MARK: - UIDeviceOrientation Extension
extension UIDeviceOrientation {
    var cgImageOrientation: CGImagePropertyOrientation {
        switch self {
        case .portrait: return .up
        case .portraitUpsideDown: return .down
        case .landscapeLeft: return .left
        case .landscapeRight: return .right
        default: return .up
        }
    }
}

// MARK: - SIMD3 Extension
extension SIMD3 where Scalar == Float {
    var allFinite: Bool {
        !x.isNaN && !y.isNaN && !z.isNaN && !x.isInfinite && !y.isInfinite && !z.isInfinite
    }
}

// MARK: - Preview
#Preview {
    ContentView()
}
