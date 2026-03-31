

<h1 align="center">🌙 LUNAR: Automated Testing of Multi-Turn Conversations in Task-Oriented LLM Systems
</h1>

## Project Structure


- [Installation](#installation)
- [Getting Started](#getting-started)
- [Replication](#replication)
- [Customization](#customization)
---

## Installation
First you need install Python 11 in your environment.

To install the LUNAR approach, create a virtual environment and install the requirements form requirements.txt in the lunar folder.

To install the baseline sensei approach follow the same steps but create an extra environment in sensei and install also its requirements there.

Then provide in .env the API KEY and endpoint to the OpenAI deployment.

To test your installation, run the script run_mt_openai.sh. Here an OpenAI based LLM is simulated for which LUNAR generated conversations.


## Getting Started

For the *Navi* case study you need first to start the open-source conversation assistant [ConvNavi](convnavi). Follow the instruction on the project website. In its .env you can select whether you want to run it in Navigation or CarControl mode.
To run long term experimetns you can run the script:

```bash
bash run_mt_navi_discrete.sh
``` 

For the *CarControl* case study run:

```bash
bash run_mt_control_discrete.sh
``` 

You can select in the script if you want to run LUNAR_S (rs flag), randomized search,
or LUNAR with optimization (ga flag).

## Replication

🚧*Instructions will be provided soon.*

### RQ1 Judge evaluation

### RQ2 Test Generation

## Customization

🚧*Instructions will be provided soon.*
