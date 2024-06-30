# LangChain Playground

This project demonstrates the use of LangChain to create an AI agent that assists in writing essays. The agent uses various prompts to plan, generate, and critique essays, leveraging the OpenAI and Tavily APIs for language generation and search functionalities, respectively.

## Project Structure
LangPlayGround/
├── .env
├── .gitignore
├── essayWriter.py
├── templates.py
└── README.md

- **.env**: Environment variables required for the project.
- **.gitignore**: Specifies files and directories to be ignored by Git.
- **essayWriter.py**: The main script that defines the essay writing agent and its workflow.
- **templates.py**: Contains prompt templates used by the agent.
- **README.md**: This README file.

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Tavily API key

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/LangPlayGround.git
   cd LangPlayGround
2.	Create and activate a virtual environment:
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
3.	Install the required packages:
pip install -r requirements.txt
4. Create a .env file in the project root directory and add your API keys:
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
## Usage
1.	Ensure you have activated your virtual environment:
Mac: source .venv/bin/activate
On Windows: .venv\Scripts\activate
2.	Run the essayWriter.py script:
python essayWriter.py
3.	The script will process the given task and output the results to the console.

Script Overview

essayWriter.py

This script defines the workflow of the essay writing agent using LangChain’s StateGraph. The agent follows these steps:

	1.	Planning: Generates a plan for the essay based on the given task.
	2.	Research: Uses Tavily to search for relevant content based on the plan.
	3.	Generation: Creates a draft of the essay using the gathered content.
	4.	Reflection: Critiques the draft to identify areas for improvement.
	5.	Research for Critique: Searches for additional content to address the critique points.
	6.	Revision: Generates a revised draft based on the critique and additional research.

templates.py

Contains the prompt templates used by the agent:

	•	PLAN_PROMPT: Template for generating an essay plan.
	•	WRITER_PROMPT: Template for generating an essay draft.
	•	REFLECTION_PROMPT: Template for critiquing the essay draft.
	•	RESEARCH_CRITIQUE_PROMPT: Template for researching additional content based on critique points.
	•	RESEARCH_PLAN_PROMPT: Template for initial research based on the essay plan.

Handling Rate Limits

The script includes a function call_openai_api_with_backoff to handle OpenAI API rate limits using exponential backoff. If the rate limit is exceeded, the script waits and retries the request.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

Acknowledgements

	•	OpenAI for their powerful language models.
	•	Tavily for their search API.
	•	LangChain for the framework used to build this agent.

 Make sure to update the placeholders (like the repository URL, API keys, etc.) with the actual information relevant to your project. This `README.md` file should provide a clear and concise guide for anyone looking to understand and use your LangChain Playground project.
 
