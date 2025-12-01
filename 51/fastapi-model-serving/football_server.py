import litserve as ls
import onnxruntime as rt
import numpy as np

# If you want to keep using your Pydantic schemas, you *can* import them:
# from schemas import FantasyAcquisitionFeatures, PredictionOutput


class InferenceEngine(ls.LitAPI):
    def setup(self, device):
        # This runs once at startup — same as your global session setup

        # Load the ONNX models (CPU like your FastAPI version)
        providers = ["CPUExecutionProvider"]
        self.sess_10 = rt.InferenceSession("acquisition_model_10.onnx",
                                           providers=providers)
        self.sess_50 = rt.InferenceSession("acquisition_model_50.onnx",
                                           providers=providers)
        self.sess_90 = rt.InferenceSession("acquisition_model_90.onnx",
                                           providers=providers)

        # Cache input/output names
        self.input_name_10 = self.sess_10.get_inputs()[0].name
        self.label_name_10 = self.sess_10.get_outputs()[0].name

        self.input_name_50 = self.sess_50.get_inputs()[0].name
        self.label_name_50 = self.sess_50.get_outputs()[0].name

        self.input_name_90 = self.sess_90.get_inputs()[0].name
        self.label_name_90 = self.sess_90.get_outputs()[0].name

    def predict(self, request):
        """
        `request` is the parsed JSON body from POST /predict.

        Expecting something like:
        {
          "waiver_value_tier": 1,
          "fantasy_regular_season_weeks_remaining": 6,
          "league_budget_pct_remaining": 37
        }
        """

        # Extract features from the JSON body
        waiver_value_tier = request["waiver_value_tier"]
        weeks_remaining = request["fantasy_regular_season_weeks_remaining"]
        budget_pct_remaining = request["league_budget_pct_remaining"]

        # Same NumPy array shape & dtype as your FastAPI code
        input_data = np.array(
            [[waiver_value_tier, weeks_remaining, budget_pct_remaining]],
            dtype=np.int64,
        )

        # Perform ONNX inference (same as your FastAPI function)
        pred_onx_10 = self.sess_10.run(
            [self.label_name_10], {self.input_name_10: input_data}
        )[0]
        pred_onx_50 = self.sess_50.run(
            [self.label_name_50], {self.input_name_50: input_data}
        )[0]
        pred_onx_90 = self.sess_90.run(
            [self.label_name_90], {self.input_name_90: input_data}
        )[0]

        # Return output of models
        return {
            "winning_bid_10th_percentile": round(float(pred_onx_10[0]), 2),
            "winning_bid_50th_percentile": round(float(pred_onx_50[0]), 2),
            "winning_bid_90th_percentile": round(float(pred_onx_90[0]), 2),
        }


if __name__ == "__main__":
    # Same LitServe boilerplate as the example
    server = ls.LitServer(InferenceEngine(max_batch_size=1), accelerator="auto")
    server.run(port=8000)
