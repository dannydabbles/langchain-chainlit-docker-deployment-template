"""Python file to serve as the frontend"""
from langchain.agents import Tool, AgentExecutor, OpenAIMultiFunctionsAgent
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory, ReadOnlySharedMemory
from langchain.tools import DuckDuckGoSearchRun
from langchain.schema.messages import SystemMessage, HumanMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.utilities.dalle_image_generator import DallEAPIWrapper
from langchain.tools import BaseTool
import random
import aiofiles
import chainlit as cl


HUMAN_PREFIX = "Protagonist"
AI_PREFIX = "Game Master"

class HumanInputChainlit(BaseTool):
    """Tool that adds the capability to ask user for input."""

    name = HUMAN_PREFIX
    description = (
        f"You can ask the {HUMAN_PREFIX} for guidance when you think you "
        "got stuck or you are not sure what to do next. "
        f"The input should be a question for the {HUMAN_PREFIX}."
    )

    def _run(
        self,
        query: str,
        run_manager=None,
    ) -> str:
        """Use the Human input tool."""

        res = run_sync(cl.AskUserMessage(content=query).send())
        return res["content"]

    async def _arun(
        self,
        query: str,
        run_manager=None,
    ) -> str:
        """Use the Human input tool."""
        res = await cl.AskUserMessage(content=query).send()
        return res["content"]

class StreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.msg = cl.Message(content="")

    async def on_llm_new_token(self, token: str, **kwargs):
        if not token:
            return
        await self.msg.stream_token(token)

    async def on_llm_end(self, response, **kwargs):
        await self.msg.send()
        self.msg = cl.Message(content="")

async def roll_dice(x: int, y: int) -> int:
    """Roll x dice with y sides each"""

    result = sum([random.randint(1, y) for _ in range(x)])

    await cl.Message(content=f"Rolled {x}d{y} and got {result}.").send()

    return result

async def roll_dice_parser(input_str: str) -> str:
    """Parse a string for dice rolls"""

    words = input_str.split()
    dice = []
    for word in words:
        # If word does not start with a number
        if word[0].isdigit():
            dice.append(word)

    # Roll all dice
    results = []
    max_result = 0
    if not dice:
        x, y = 1, 20
        results.append(await roll_dice(int(x), int(y)))
        max_result += int(x) * int(y)
    for die in dice:
        try:
            x, y = die.split("d")
        except ValueError:
            x, y = 1, 20
        results.append(await roll_dice(int(x), int(y)))
        max_result += int(x) * int(y)

    result = sum(results)

    message = f"Rolled {result} out of a possible {max_result}."

    await cl.Message(content=message).send()

    return message

async def generate_dalle_image(text):
    """Generate an image with DALL-E for the given text based on the conversation history thus far"""

    print(f"Generating DALL-E image for: {text}")

    image_url = DallEAPIWrapper().run(text)

    elements = [
        cl.Image(name=text, display="inline", url=image_url)
    ]

    await cl.Message(content=text, elements=elements).send()
    
def get_memory(llm):
    """Create a memory buffer for the conversation"""

    memory_buffer = ConversationSummaryBufferMemory(
        llm=llm,
        memory_key="chat_history",
        max_token_limit=12000,
        ai_prefix=AI_PREFIX,
        human_prefix=HUMAN_PREFIX,
        input_key="input",
        return_messages=True,
    )

    return memory_buffer

def get_tools(llm):
    """Create a list of tools"""

    tools_data = {
        "Dice_Rolling_Assistant": {
            "description": "An assistant that can roll any combination of dice. Useful for adding unpredictability and uniqueness to story elements. Dice roll results represent the positive (high) or negative (low) outcomes of events against a predetermined difficulty value. The input to this tool should follow the exact format XdY where X is the number of dice, and Y is the number of sides on each die rolled (e.g. 4d20, or 1d6).",
            "coroutine": roll_dice_parser,
        },
        "Internet_Search": {
            "description": "Useful for looking up unknown information about D&D 5e rules, lore, and other fine details. The input to this tool should be a google search query. Ask targeted questions! The result of this tool should be a summary of the search results.",
            "coroutine": DuckDuckGoSearchRun().arun,
        },
        "Generate_Image_URL_DALL-E": {
            "description": """Useful for generating DALL-E Image URLs to match a story element. The input to this tool should be a detailed specific prompt that DALL-E can use to show the story element. The result of this tool should be a cinematic well framed image generated by DALL-E to represent what's happening in the story.
""",
            "coroutine": generate_dalle_image,
        },
    }

    tools = [HumanInputChainlit()]
    for tool_name, tool_info in tools_data.items():
        description = tool_info["description"]

        if 'function' in tool_info:
            func = tool_info['function']
            tool = Tool(name=tool_name, func=func, coroutine=None, description=description)
        elif 'coroutine' in tool_info:
            coroutine = tool_info['coroutine']
            tool = Tool(name=tool_name, func=None, coroutine=coroutine, description=description)
        else:
            raise ValueError("Tool must have a function or coroutine")

        tools.append(tool)

    return tools

def get_conversation_chain(llm, memory, tools):
    """Create a chain of agents for the conversation"""

    prompt = OpenAIMultiFunctionsAgent.create_prompt(
        system_message=SystemMessage(content="""Never forget you are the Storyteller, and I am the protagonist. You are an experienced Game Master playing a homebrew tabletop story with your new friend, me. I will propose actions I plan to take and you will explain what happens when I take those actions. Speak in the first person from the perspective of the Game Master, and generate lots of images inline. For describing actions that happen in the game world, wrap each description word block in '*' characters.
Do not shorten Image URLs *at all*, or the image will not show up!
Do not change roles unless acting out a character besides the protagonist!
Do not *ever* speak from the perspective of the protagonist!
Do not add anything else.
Remember you are the storyteller and Game Master, and I am the protagonist.
Stop speaking the moment you finish speaking from your perspective."""),
        extra_prompt_messages=[
            MessagesPlaceholder(variable_name="chat_history"),
        ],
    )

    agent = OpenAIMultiFunctionsAgent(
        llm=llm,
        tools=tools,
        prompt=prompt,
        memory=ReadOnlySharedMemory(memory=memory),
        verbose=True,
    )

    agent_chain = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory,
        handle_parsing_errors="As a Storyteller, that doesn't quite make sense.  What else can I try?",
        max_iterations=5,
        max_execution_time=120,
    )

    return agent_chain

@cl.on_chat_start
def start():
    """Start the conversation"""

    llmc = ChatOpenAI(temperature=.7, streaming=True, model="gpt-3.5-turbo-16k")

    memory = get_memory(llmc)
    tools = get_tools(llmc)
    agent_chain = get_conversation_chain(llmc, memory, tools)

    cl.user_session.set("agent_chain", agent_chain)
    cl.user_session.set("memory", memory)
    cl.user_session.set("tools", tools)
    cl.user_session.set("llmc", llmc)

@cl.on_message
async def main(message):
    """Main function"""

    # Get the agent chain
    agent_chain = cl.user_session.get("agent_chain")
    llmc = cl.user_session.get("llmc")

    # Call the agent chain
    content = str(message.content)
    content += "\n\nGenerate a DALL-E image url too, but don't show it to me!"
    reply = await agent_chain.acall(
        {
            "input": content,
        },
        callbacks=[
            cl.AsyncLangchainCallbackHandler(stream_final_answer=True),
            StreamHandler()
        ]
    )
    print(f"Reply: {reply}")

    # Prune memory
    agent_chain.memory.prune()

