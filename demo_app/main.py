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
import random
import aiofiles
import chainlit as cl


HUMAN_PREFIX = "Protagonist"
AI_PREFIX = "Game Master"

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
        results.append(roll_dice(int(x), int(y)))
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

def get_tools(llm, memory):
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
    }

    tools = []
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
        system_message=SystemMessage(content="""Never forget you are the Storyteller, and I am the protagonist. You are an experienced Game Master playing a homebrew tabletop story with your new friend, me. Never tell me your dice rolls! I will propose actions I plan to take and you will explain what happens when I take those actions. Speak in the first person from the perspective of the Game Master. For describing actions that happen in the game world, wrap each description word block in '*' characters. The success of my actions should be determined by a dice roll where appropriate by D&D 5e rules!
Do not change roles unless acting out a character besides the protagonist!
Do not *ever* speak from the perspective of the protagonist!
Do not add anything else
Remember you are the storyteller and Game Master.
Stop speaking the moment you finish speaking from your perspective as the Game Master."""),
        extra_prompt_messages=[
            MessagesPlaceholder(variable_name="chat_history"),
        ],
    )

    agent = OpenAIMultiFunctionsAgent(
        llm=llm,
        tools=tools,
        prompt=prompt,
        memory=memory,
        verbose=True,
        max_execution_time=60,
    )

    agent_chain = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory,
        handle_parsing_errors="As a Storyteller, that doesn't quite make sense.  What else can I try?",
        max_iterations=5,
    )

    return agent_chain

@cl.on_chat_start
def start():
    """Start the conversation"""

    llmc = ChatOpenAI(temperature=.7, streaming=True, model="gpt-3.5-turbo-16k")

    memory = get_memory(llmc)
    tools = get_tools(llmc, memory)
    agent_chain = get_conversation_chain(llmc, memory, tools)

    cl.user_session.set("agent_chain", agent_chain)
    cl.user_session.set("memory", memory)
    cl.user_session.set("tools", tools)
    cl.user_session.set("llmc", llmc)

def get_dalle_template(memory) -> PromptTemplate:
    """Create a template for DALL-E prompts"""

    chat_history = ""
    for engram in memory.chat_memory.messages:
        if engram.type == "human":
            chat_history += f"{HUMAN_PREFIX}: {engram.content}\n"
        elif engram.type == "ai":
            chat_history += f"{AI_PREFIX}: {engram.content}\n"

    template_start = "{input}\n\n"
    template_history = f"""Conversation History:
{chat_history}
"""
    template = template_start + template_history

    prompt = PromptTemplate(
        template=template,
        input_variables=[],
    )

    return prompt

def generate_dalle_image(llm, memory):
    """Generate an image with DALL-E for the given text based on the conversation history thus far"""

    # Make the prompt for DALL-E
    prompt = get_dalle_template(memory)

    # Make a LLMChain to write a prompt for DALL-E based on the text
    dalle_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=ReadOnlySharedMemory(memory=memory),
        verbose=True,
    )

    # Run the LLM chain
    dalle_prompt_input = """Compose a DALL-E prompt based on the conversation history thus far. The resulting image should be cinematic and represent the current state of the story. Include as much detail as DALL-E can handle, but be concise."""
    dalle_prompt = dalle_chain.run(dalle_prompt_input)

    print(f"{dalle_prompt}")

    image_url = DallEAPIWrapper().run(dalle_prompt)

    return image_url, dalle_prompt
    
class StreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.msg = cl.Message(content="")

    async def on_llm_new_token(self, token: str, **kwargs):
        if not token:
            return
        await self.msg.stream_token(token)

    async def on_llm_end(self, response: str, **kwargs):
        await self.msg.send()
        self.msg = cl.Message(content="")

@cl.on_message
async def main(message):
    """Main function"""

    # Get the agent chain
    agent_chain = cl.user_session.get("agent_chain")
    llmc = cl.user_session.get("llmc")

    # Call the agent chain
    reply = await agent_chain.acall(
        {
            "input": message.content,
        },
        callbacks=[
            cl.AsyncLangchainCallbackHandler(stream_final_answer=True),
            StreamHandler()
        ]
    )
    print(f"Reply: {reply}")

    image_url, dalle_prompt = generate_dalle_image(llmc, agent_chain.memory)

    elements = [
        cl.Image(name="Game Frame", display="inline", url=image_url),
    ]

    await cl.Message(content="Image:", elements=elements).send()

    # Prune memory
    agent_chain.memory.prune()

