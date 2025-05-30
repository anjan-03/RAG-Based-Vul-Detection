The workflow starts by running the codebase through a static analysis tool called CodeQL. 

This helps identify possible vulnerabilities in the code through its queries. Next, a Software Bill of Materials (SBOM) is generated, which lists all the packages and dependencies used in the project. Along with the SBOM, general documentation about packages, typically in PDF form, is also collected. 

Both the SBOM and the documentation are then fed into a Retrieval-Augmented Generation (RAG) model-specifically, a Mistral model running via Ollama. This model analyzes the information and predicts the types of vulnerabilities that might be present.
