# 🧮 sommenGeneratorServer

Dit is de backend-server voor de educatieve sommen-generator. De server maakt gebruik van Azure OpenAI om verhaaltjessommen te genereren op basis van input van de gebruiker.

## ⚙️ Setup

### 1. Repository klonen

```bash
git clone https://github.com/jouwgebruikersnaam/sommenGeneratorServer.git
cd sommenGeneratorServer
````

### 2. Dependencies installeren

npm install

### 3. .env bestand aanmaken

```bash
AZURE_OPENAI_API_VERSION=2025-03-01-preview
AZURE_OPENAI_API_INSTANCE_NAME=cmgt-ai
AZURE_OPENAI_API_KEY=VUL_HIER_JE_EIGEN_API_KEY_IN
AZURE_OPENAI_API_DEPLOYMENT_NAME=deploy-gpt-35-turbo
AZURE_OPENAI_API_EMBEDDINGS_DEPLOYMENT_NAME=deploy-text-embedding-ada
```

### 4. Server starten
npm run dev
