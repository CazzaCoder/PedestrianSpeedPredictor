import SwiftUI
import RealityKit
import ARKit
import Vision
import CoreML

struct ContentView: View {
    var body: some View {
        ARViewContainer()
            .edgesIgnoringSafeArea(.all)
    }
}

struct ARViewContainer: UIViewRepresentable {
    func makeUIView(context: Context) -> CustomARView {
        let arView = CustomARView(frame: .zero)
        return arView
    }
    
    func updateUIView(_ uiView: CustomARView, context: Context) {}
}

// Represents a tracked person with their data
class PersonTrack {
    let id: UUID
    var lastObservation: VNDetectedObjectObservation
    var position: SIMD3<Float>?
    var lastUpdateTime: TimeInterval
    var anchor: AnchorEntity? // Store anchor for position updates
    var entity: ModelEntity?  // Speed label entity
    var speed: Float = 0.0
    var direction: SIMD3<Float> = .zero
    var lastConfidence: Float = 0.0
    var arrowEntity: ModelEntity?
    
    init(id: UUID, observation: VNDetectedObjectObservation, time: TimeInterval) {
        self.id = id
        self.lastObservation = observation
        self.lastUpdateTime = time
    }
    
    // Cleanup method to remove from scene
    func removeFromScene(_ scene: RealityKit.Scene) {
        if let anchor = anchor {
            scene.removeAnchor(anchor)
        }
    }
}

class CustomARView: ARView {
    // Vision & Model
    private var visionRequests = [VNRequest]()
    private let model = try! VNCoreMLModel(for: yolo11n_1080(configuration: MLModelConfiguration()).model)
    
    // Tracking
    private var tracks: [PersonTrack] = []
    private var frameCounter = 0
    private let detectionInterval = 3 // Run detection every 5 frames for performance
    private var lastFrameTimestamp: CFTimeInterval = 0.0
    
    required init(frame frameRect: CGRect) {
        super.init(frame: frameRect)
        
        // Setup AR configuration
        let config = ARWorldTrackingConfiguration()
        config.planeDetection = [.horizontal]
        session.run(config)
        
        setupVision()
        session.delegate = self
    }
    
    @objc required dynamic init?(coder decoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    private func setupVision() {
        let request = VNCoreMLRequest(model: model) { request, error in
            guard let observations = request.results as? [VNRecognizedObjectObservation] else { return }
            let personObservations = observations.filter { $0.labels.first?.identifier.lowercased() == "person" && $0.confidence > 0.5 }
            self.handleNewDetections(personObservations)
        }
        request.imageCropAndScaleOption = .scaleFill
        visionRequests = [request]
    }
    
    // Handle new detections and associate with tracks
    private func handleNewDetections(_ detections: [VNRecognizedObjectObservation]) {
        print("[handleNewDetections] Received \(detections.count) detections")
        var newTracks: [PersonTrack] = []
        let currentTime = CACurrentMediaTime()
        
        for detection in detections {
            var matched = false
            for track in tracks {
                let iou = computeIoU(box1: detection.boundingBox, box2: track.lastObservation.boundingBox)
                if iou > 0.5 { // Threshold for matching
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
    
    // Compute Intersection over Union for bounding box association
    private func computeIoU(box1: CGRect, box2: CGRect) -> Float {
        let intersection = box1.intersection(box2)
        if intersection.isNull { return 0.0 }
        let intersectionArea = intersection.width * intersection.height
        let unionArea = (box1.width * box1.height) + (box2.width * box2.height) - intersectionArea
        return Float(intersectionArea / unionArea)
    }
    
    // Update track position and speed
    private func updateTrackPositionAndSpeed(track: inout PersonTrack, frame: ARFrame) {
        let boundingBox = track.lastObservation.boundingBox
        let centerVision = CGPoint(x: boundingBox.midX, y: boundingBox.minY) // Bottom center for feet
        let viewSize = self.frame.size
        let centerPixel = CGPoint(x: centerVision.x * viewSize.width, y: (1 - centerVision.y) * viewSize.height)
        
        guard let rayResult = ray(through: centerPixel) else { return }
        
        // Raycast to horizontal planes
        let query = ARRaycastQuery(origin: rayResult.origin, direction: rayResult.direction, allowing: .estimatedPlane, alignment: .horizontal)
        if let result = session.raycast(query).first {
            let newPosition = SIMD3(result.worldTransform.columns.3.x, result.worldTransform.columns.3.y, result.worldTransform.columns.3.z)
            
            // Compute direction vector and apply smoothing.
            if let oldPosition = track.position {
                let rawDirection = newPosition - oldPosition
                // Use a small alpha for smoothing
                let alpha: Float = 0.7
                track.direction = alpha * track.direction + (1 - alpha) * simd_normalize(rawDirection)
            }
            
            if let oldPosition = track.position, frame.timestamp > track.lastUpdateTime {
                let dt = Float(frame.timestamp - track.lastUpdateTime)
                if dt > 0 {
                    let distance = simd_length(newPosition - oldPosition)
                    track.speed = distance / dt
                }
            }
            track.position = newPosition
            track.lastUpdateTime = frame.timestamp
            updateEntityForTrack(&track)
        }
    }
    
    // Create or update speed display entity
    private func updateEntityForTrack(_ track: inout PersonTrack) {
        let speedString = String(format: "%.2f m/s", track.speed)
        
        if let anchor = track.anchor, let entity = track.entity, let position = track.position {
            // Update existing entity
            anchor.position = position // Update anchor to new base position
            entity.position = SIMD3<Float>(0, 1.5, 0) // 1.5m above anchor
            let newMesh = MeshResource.generateText(speedString, font: .systemFont(ofSize: 0.1), containerFrame: .zero, alignment: .center, lineBreakMode: .byWordWrapping)
            entity.model?.mesh = newMesh
            
            // Create or update the arrow entity
            let arrowLength = track.speed * 0.5 // predicts half a second distance
            let arrowDirection = track.direction
            
            // If arrowEntity doesn't exist, create it
            if track.arrowEntity == nil {
                // We'll build a simple arrow from a cylinder + cone.
                let shaftRadius: Float = 0.02
                let shaftHeight: Float = 1.0
                let coneHeight: Float = 0.2

                let shaftMesh = MeshResource.generateCylinder(height: shaftHeight, radius: shaftRadius)
                let coneMesh = MeshResource.generateCone(height: coneHeight, radius: 0.05)

                let shaftEntity = ModelEntity(mesh: shaftMesh, materials: [SimpleMaterial(color: .blue, isMetallic: false)])
                // The cylinder is centered on its local origin, so we move it up by half
                shaftEntity.position.y = shaftHeight / 2

                let coneEntity = ModelEntity(mesh: coneMesh, materials: [SimpleMaterial(color: .blue, isMetallic: false)])
                // Place cone at top of cylinder
                coneEntity.position.y = shaftHeight

                // Combine into a parent entity
                let arrowParent = ModelEntity()
                arrowParent.addChild(shaftEntity)
                arrowParent.addChild(coneEntity)

                // Scale it down initially
                arrowParent.scale = SIMD3<Float>(repeating: 0.001)

                // Attach to anchor
                track.anchor?.addChild(arrowParent)
                track.arrowEntity = arrowParent
            }

            if let arrowEntity = track.arrowEntity {
                // If confidence has dipped below 0.5, reduce opacity
                let arrowOpacity: Float = track.lastConfidence < 0.5 ? 0.3 : 1.0
                let materials = [SimpleMaterial(color: .blue.withAlphaComponent(CGFloat(arrowOpacity)), isMetallic: false)]

                // Update materials on arrow children
                for child in arrowEntity.children {
                    if var model = child as? ModelEntity {
                        model.model?.materials = materials
                    }
                }

                // Adjust total arrow length to reflect speed * time window.
                // We'll scale the arrow's height accordingly.
                // The default arrow is 1.2m tall (cylinder + cone). We'll scale it.
                let baseHeight: Float = 1.2
                let targetScale = arrowLength / baseHeight

                // Update arrow scale
                arrowEntity.scale = SIMD3<Float>(repeating: 0.001 + max(targetScale, 0.0))

                // Rotate the arrow to point in track.direction
                // The arrow is oriented along +Y, so we compute a rotation from +Y to arrowDirection.
                // If arrowDirection is near zero, skip.
                if simd_length(arrowDirection) > 0.0001 {
                    // Normalized direction in XZ plane or full 3D?
                    let up = SIMD3<Float>(0, 1, 0)
                    // Axis is cross( up, arrowDirection ), angle is arccos(dot(up, arrowDirection)).
                    let dirNorm = simd_normalize(arrowDirection)
                    let dotVal = simd_dot(up, dirNorm)
                    let angle = acos(dotVal)
                    let axis = simd_normalize(simd_cross(up, dirNorm))
                    if !axis.x.isNaN && !axis.y.isNaN && !axis.z.isNaN {
                        arrowEntity.orientation = simd_quatf(angle: angle, axis: axis)
                    }
                }

                // Place arrow at feet level. The anchor is at the person's foot position, so we just do:
                arrowEntity.position = .zero
            }
        } else if let position = track.position {
            // Create new anchor and entity
            let anchor = AnchorEntity(world: position)
            let labelEntity = ModelEntity(
                mesh: MeshResource.generateText(speedString, font: .systemFont(ofSize: 0.1), containerFrame: .zero, alignment: .center, lineBreakMode: .byWordWrapping),
                materials: [SimpleMaterial(color: .white, isMetallic: false)]
            )
            labelEntity.scale = SIMD3<Float>(repeating: 0.2)
            labelEntity.position = SIMD3<Float>(0, 1.5, 0) // 1.5m above anchor
            anchor.addChild(labelEntity)
            scene.addAnchor(anchor)
            track.anchor = anchor
            track.entity = labelEntity
            
            // Create or update the arrow entity
            let arrowLength = track.speed * 0.5 // predicts half a second distance
            let arrowDirection = track.direction
            
            // If arrowEntity doesn't exist, create it
            if track.arrowEntity == nil {
                // We'll build a simple arrow from a cylinder + cone.
                let shaftRadius: Float = 0.02
                let shaftHeight: Float = 1.0
                let coneHeight: Float = 0.2

                let shaftMesh = MeshResource.generateCylinder(height: shaftHeight, radius: shaftRadius)
                let coneMesh = MeshResource.generateCone(height: coneHeight, radius: 0.05)

                let shaftEntity = ModelEntity(mesh: shaftMesh, materials: [SimpleMaterial(color: .blue, isMetallic: false)])
                // The cylinder is centered on its local origin, so we move it up by half
                shaftEntity.position.y = shaftHeight / 2

                let coneEntity = ModelEntity(mesh: coneMesh, materials: [SimpleMaterial(color: .blue, isMetallic: false)])
                // Place cone at top of cylinder
                coneEntity.position.y = shaftHeight

                // Combine into a parent entity
                let arrowParent = ModelEntity()
                arrowParent.addChild(shaftEntity)
                arrowParent.addChild(coneEntity)

                // Scale it down initially
                arrowParent.scale = SIMD3<Float>(repeating: 0.001)

                // Attach to anchor
                track.anchor?.addChild(arrowParent)
                track.arrowEntity = arrowParent
            }

            if let arrowEntity = track.arrowEntity {
                // If confidence has dipped below 0.5, reduce opacity
                let arrowOpacity: Float = track.lastConfidence < 0.5 ? 0.3 : 1.0
                let materials = [SimpleMaterial(color: .blue.withAlphaComponent(CGFloat(arrowOpacity)), isMetallic: false)]

                // Update materials on arrow children
                for child in arrowEntity.children {
                    if var model = child as? ModelEntity {
                        model.model?.materials = materials
                    }
                }

                // Adjust total arrow length to reflect speed * time window.
                // We'll scale the arrow's height accordingly.
                // The default arrow is 1.2m tall (cylinder + cone). We'll scale it.
                let baseHeight: Float = 1.2
                let targetScale = arrowLength / baseHeight

                // Update arrow scale
                arrowEntity.scale = SIMD3<Float>(repeating: 0.001 + max(targetScale, 0.0))

                // Rotate the arrow to point in track.direction
                // The arrow is oriented along +Y, so we compute a rotation from +Y to arrowDirection.
                // If arrowDirection is near zero, skip.
                if simd_length(arrowDirection) > 0.0001 {
                    // Normalized direction in XZ plane or full 3D?
                    let up = SIMD3<Float>(0, 1, 0)
                    // Axis is cross( up, arrowDirection ), angle is arccos(dot(up, arrowDirection)).
                    let dirNorm = simd_normalize(arrowDirection)
                    let dotVal = simd_dot(up, dirNorm)
                    let angle = acos(dotVal)
                    let axis = simd_normalize(simd_cross(up, dirNorm))
                    if !axis.x.isNaN && !axis.y.isNaN && !axis.z.isNaN {
                        arrowEntity.orientation = simd_quatf(angle: angle, axis: axis)
                    }
                }

                // Place arrow at feet level. The anchor is at the person's foot position, so we just do:
                arrowEntity.position = .zero
            }
        }
    }
}

// MARK: - ARSessionDelegate
extension CustomARView: ARSessionDelegate {
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let deltaT = frame.timestamp - lastFrameTimestamp
        print("[INFO] Frame Time Delta: \(String(format: "%.3f", deltaT)) seconds")
        lastFrameTimestamp = frame.timestamp
        autoreleasepool {
            frameCounter += 1
            let pixelBuffer = frame.capturedImage
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
            
            var requests: [VNRequest] = []
            
            // Periodic detection
            if frameCounter % detectionInterval == 0 {
                requests.append(visionRequests[0])
            }
            
            // Tracking for existing tracks
            for track in tracks {
                let trackingRequest = VNTrackObjectRequest(detectedObjectObservation: track.lastObservation)
                trackingRequest.trackingLevel = .accurate
                requests.append(trackingRequest)
            }
            
            do {
                try handler.perform(requests)
                print("[INFO] Frame #\(frameCounter) processed at timestamp \(frame.timestamp)")
                
                // Handle detection results
                if frameCounter % detectionInterval == 0 {
                    print("Calculating Detection...")
                    guard let detectionRequest = requests.first(where: { $0 is VNCoreMLRequest }) as? VNCoreMLRequest,
                          let observations = detectionRequest.results as? [VNRecognizedObjectObservation] else { return }
                    let personObservations = observations.filter { $0.labels.first?.identifier.lowercased() == "person" && $0.confidence > 0.5 }
                    handleNewDetections(personObservations)
                    print("[INFO] Number of recognized observations: \(observations.count)")
                    for obs in observations {
                    let bestLabel = obs.labels.first?.identifier ?? "unknown"
                    let conf = obs.confidence
                    print("   Detected \(bestLabel) with confidence \(String(format: "%.2f", conf))")
                    }
                }
                
                // Handle tracking results
                for request in requests where request is VNTrackObjectRequest {
                    if let result = request.results?.first as? VNDetectedObjectObservation {
                        if let index = tracks.firstIndex(where: { $0.lastObservation == (request as! VNTrackObjectRequest).inputObservation }) {
                            tracks[index].lastObservation = result
                            updateTrackPositionAndSpeed(track: &tracks[index], frame: frame)
                        }
                    }
                }
                
                // Clean up old tracks
                let currentTime = frame.timestamp
                let activeTracks = tracks.filter { currentTime - $0.lastUpdateTime < 1.0 } // Tracks expire after 1 second
                for track in tracks where !activeTracks.contains(where: { $0.id == track.id }) {
                    track.removeFromScene(scene) // Properly remove anchor from scene
                }
                tracks = activeTracks
                
            } catch {
                print("Failed to perform Vision requests: \(error)")
            }
        }
    }
}

#Preview {
    ContentView()
}
