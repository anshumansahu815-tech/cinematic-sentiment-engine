🎬 Cinematic Sentiment Engine
Tagline: Moving beyond binary "Good/Bad" with a 28-class RoBERTa neural network that translates nuanced audience reviews into actionable studio insights.

💡 Inspiration (The Problem)
In the film industry, audience feedback is everything. But when we tested standard, out-of-the-box AI sentiment analyzers, we noticed a massive flaw: they completely fail to understand human nuance.

If an audience member writes, "The visual effects were absolutely mind-blowing, but the systemic corruption subplot felt heavy-handed," a standard NLP model will flag it as 99.9% NEGATIVE simply because it sees words like "corruption" and "heavy-handed." It entirely misses the praise. We realized that binary (Positive/Negative) classification is practically useless for real-world studio executives who need to understand complex, mixed audience reactions.

⚙️ What it does (The Solution)
The Cinematic Sentiment Engine is a context-aware AI dashboard built for the entertainment industry. Instead of basic binary classification, it utilizes a highly advanced 28-class emotion detection model.

When a user submits a review, the engine:

Identifies the Dominant Emotion: (e.g., Admiration, Amusement, Disappointment).

Provides a Nuance Breakdown: It extracts the top three conflicting emotions to show the layers of the audience's reaction.

Generates Studio Recommendations: It translates raw mathematical probabilities into actionable business logic (e.g., advising a studio to "Highlight quote in marketing materials" or "Flag pacing issues for the writers' room").

🛠️ How we built it
Frontend: We built a commercial-grade, responsive UI using Streamlit, featuring a two-column layout, simulated processing telemetry, and dynamic color-coded metric bars.

Machine Learning Backend: We leveraged the Hugging Face Transformers library, specifically implementing the SamLowe/roberta-base-go_emotions pipeline.

The "Judge-Breaker" Logic: By manipulating the top_k=3 parameters of the pipeline, we forced the model to return a multi-dimensional array of emotions, which we then parsed into a custom business-logic engine.

🚧 Challenges we ran into
The biggest hurdle was overcoming the limitations of pre-trained binary classifiers. Our initial prototype successfully ran a standard DistilBERT model, but the results were too flat to be useful. We had to pivot our entire backend strategy mid-hackathon to integrate a multi-label RoBERTa architecture that could actually comprehend contextual subtext without breaking our frontend UI. We also overcame local environment configurations and Git deployment hurdles to get the project version-controlled and live!

🏆 Accomplishments that we're proud of
We are incredibly proud of transforming a raw Python script into a fully realized, polished web app. Adding the "Studio Recommendation" feature bridged the gap between pure machine learning math and real-world business value. It doesn't just show data; it makes decisions.

🚀 What's next for the Cinematic Sentiment Engine
In the future, we want to scale this application to scrape live Twitter/X feeds and Rotten Tomatoes pages via API. This would allow a studio to monitor the "Nuance Breakdown" of thousands of live reactions on opening night in real-time, giving them an unprecedented dashboard for audience retention.