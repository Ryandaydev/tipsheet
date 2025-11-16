import litserve as ls

# define the api to include any number of models, dbs, etc...
class InferenceEngine(ls.LitAPI):
    def setup(self, device):
        self.text_model = lambda x: x**2
        self.vision_model = lambda x: x**3

    def predict(self, request):
        x = request["input"]    
        # perform calculations using both models
        a = self.text_model(x)
        b = self.vision_model(x)
        c = a + b
        return {"output": c}

if __name__ == "__main__":
    # 12+ features like batching, streaming, etc...
    server = ls.LitServer(InferenceEngine(max_batch_size=1), accelerator="auto")
    server.run(port=8000)