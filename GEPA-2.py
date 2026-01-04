class GEPA:
    def __init__(self, llm_call):
        self.llm = llm_call
        self.prompt_memory=[]

    def generate(self, prompt):
        response = self.llm(prompt)
        return int(response.strip())

    def evaluate(self, pred, mean, std):
        dist = abs(pred - mean)
        reward = max(0, 1 - dist / 4)
        if dist <= std:
            reward += 0.25
        return reward
    def reflect(self, pred, mean, std):
        reflection_prompt = f"""
            You predicted {pred}.
            Human mean: {mean}
            Human std: {std}
            """
         return self.llm(reflection_prompt)

    def adapt_prompt(self, base_prompt, reflection):
        return base_prompt + "\n\nGuidance based on prior errors:\n" + reflection

#Training loop
def run_gepa(sample, base_prompt, gepa, steps=3):
    prompt = base_prompt

    for _ in range(steps):
        pred = gepa.generate(prompt)
        reward = gepa.evaluate(pred, sample["average"], sample["std"])
        reflection = gepa.reflect(pred, sample["average"], sample["std"])
        prompt = gepa.adapt_prompt(prompt, reflection)

    return prompt

