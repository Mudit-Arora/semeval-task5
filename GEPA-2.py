class GEPA:
    def __init__(self, llm_call):
        self.llm = llm_call
        self.prompt_memory = []

    def generate(self, prompt):
        response = self.llm(prompt)
        return int(response.strip())

    def evaluate(self, pred, mean, std):
        dist = abs(pred - mean)
        reward = max(0, 1 - dist / 4)
        if dist <= std:
            reward += 0.25
        return reward
