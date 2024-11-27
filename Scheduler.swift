import CoreML

class ModelScheduler {
    private let queue = DispatchQueue(label: "com.ml.scheduler",
                                    qos: .userInteractive)
    
    func scheduleInference(model: MLModel, 
                          input: MLFeatureProvider) async throws -> MLFeatureProvider {
        return try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do {
                    let result = try model.prediction(from: input)
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    func batchProcess(models: [MLModel], 
                     inputs: [MLFeatureProvider]) async throws -> [MLFeatureProvider] {
       
        try await withThrowingTaskGroup(of: MLFeatureProvider.self) { group in
            for (model, input) in zip(models, inputs) {
                group.addTask {
                    try await self.scheduleInference(model: model, 
                                                   input: input)
                }
            }
            
            var results = [MLFeatureProvider]()
            for try await result in group {
                results.append(result)
            }
            return results
        }
    }
} 