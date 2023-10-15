"""Python file to serve as the frontend"""
from langchain import OpenAI # LLMMathChain SerpAPIWrapper?
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import initialize_agent, ZeroShotAgent, Tool, AgentExecutor, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationTokenBufferMemory, ConversationKGMemory, CombinedMemory, ReadOnlySharedMemory
from langchain.chains import LLMMathChain, LLMChain, ConversationChain
import os
import random
import chainlit as cl
from langchain import hub


def create_chain(chain_type, llm, input_variables, template_str, memory):

    if template_str:
        prompt = PromptTemplate(
            input_variables=input_variables,
            template=template_str,
        )

    if chain_type == LLMMathChain:
        return chain_type(
            llm=llm,
            #prompt=prompt,
            verbose=True,
            #memory=memory,
            input_key="input",
        )

    return chain_type(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=ReadOnlySharedMemory(memory=CombinedMemory(memories=memory)),
    )

def roll_dice(x: int, y: int) -> int:
    return sum([random.randint(1, y) for _ in range(x)])

def roll_dice_parser(input_str: str) -> str:
    dice_nums = [int(x.strip()) for x in input_str.split(",")]
    dice_rolls = [roll_dice(x, y) for x, y in zip(dice_nums[::2], dice_nums[1::2])]
    total = sum(dice_rolls)
    dice_description = [f"{x}d{y}" for x, y in zip(dice_nums[::2], dice_nums[1::2])]
    return f"Rolling {', '.join(dice_description)}: {', '.join([str(x) for x in dice_rolls])}. Total: {total}"

def get_memory(llm):
    memory_buffer = ConversationTokenBufferMemory(
        llm=llm,
        memory_key="buffer",
        max_tokens=1000,
        ai_prefix="Game Master",
        human_prefix="Protagonist",
        input_key="input",
    )

    memory_entity = ConversationKGMemory(
        llm=llm,
        memory_key="entities",
        input_key="input",
    )

    return [memory_buffer, memory_entity]

def get_conversation_chain(llm, memory, tools):
    prefix = """Never forget you are the Storyteller, and I am the protagonist. You are an experienced Game Master playing a homebrew tabletop story with your new friend, me. Never tell me your dice rolls! I will propose actions I plan to take and you will explain what happens when I take those actions. Speak in the first person from the perspective of the Game Master. For describing actions that happen in the game world, wrap your description in '*'.
Do not change roles unless acting out a character besides the protagonist!
Do not speak from the perspective of the protagonist.
Do not forget to finish speaking by saying, 'It's your turn, what do you do next?'
Do not add anything else
Remember you are the storyteller and Game Master.
Stop speaking the moment you finish speaking from your perspective as the Game Master.

If you need inspiration, use the tools. You have access to the following tools:"""
    suffix = """As Game Master, you have taken the following notes about the game thus far. Use these notes to decide what happens next in the story.

Protagonist's Action:
{input}

Story History:
{buffer}

Story Entities Data:
{entities}

Game Scratchpad:"""

    prompt = ZeroShotAgent.create_prompt(tools, prefix=prefix, suffix=suffix, input_variables=["input", "buffer", "entities"])
    llm_chain = ConversationChain(llm=llm, prompt=prompt, verbose=True, memory=CombinedMemory(memories=[ReadOnlySharedMemory(memory=memory[0]), memory[1]]))

    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=CombinedMemory(memories=memory),
        handle_parsing_errors="As a Storyteller, that doesn't quite make sense.  What else can I try?",
        max_iterations=3,
        early_stopping_method="generate",
    )

    return agent_chain

def get_llmmath_chain(llm, memory, tools, prompt="You are an AI Game Master for a D&D 5e campaign, what do you say next?"):

    agent = LLMMathChain(
        llm=llm,
        tools=tools,
        memory=memory,
        verbose=True,
        max_iterations=3,
        early_stopping_method="generate",
        handle_parsing_errors="As the Game Master, I'll put it a different way...",
        prompt=prompt,
        ai_prefix="Game Master",
        human_prefix="Protagonist",
    )

    return agent

def get_tools(llm, memory):
    tools_data = {
        "Make up a new story": {
            "template": """Makes up the first act of a new story. Incorporate the following information into the story:

Story Inspiration:
{input}

Campaign History:
{buffer}

Campaign Entities Data:
{entities}

Storyteller:""",
            "description": "Useful for when the Campaign History is empty. This tool will generate a new story for you to start the story. The input to this tool should be a summary of any information that should be included in the story. The result of this tool should be a summary of the first act of the generated story.",
            "input_variables": ["input", "buffer", "entities"],
            "chain_type": ConversationChain,
            "return_direct": False,
        },
        "Dice Rolling Assistant": {
            "description": "An assistant that can roll any combination of dice. Useful for adding unpredictability and uniqueness to story elements. Dice roll results represent the positive (high) or negative (low) outcomes of events against a predetermined difficulty value. The input to this tool should be a comma separated list of numbers, each pair representing the number of dice followed by the number or sides on each die. For example, '1, 6, 10, 4' would be the input if you wanted to roll 1d6 and 10d4 dice.",
            "function": roll_dice_parser,
        },
        "Internet Search": {
            "description": "Searches the internet for information, facts, or story ideas. Useful for finding inspiration for story elements. The input to this tool should be a search query. The result of this tool should be a summary of the search results. Be direct!",
            "function": cl.user_session.get("search").run,
        },
    }
    
    tools = []
    for tool_name, tool_info in tools_data.items():
        description = tool_info["description"]

        if 'function' in tool_info:
            func = tool_info['function']
            tool = Tool(name=tool_name, func=func, description=description)
        else:
            template_str = tool_info.get("template", None)
            input_variables = tool_info["input_variables"]
            chain_type = tool_info["chain_type"]
            return_direct = tool_info.get("return_direct", False)
            cl.user_session.set("chain_type", chain_type)
            chain = create_chain(chain_type, llm, input_variables, template_str, memory)
            tool = Tool(name=tool_name, func=chain.run, description=description, return_direct=return_direct)
        tools.append(tool)

    return tools

@cl.on_chat_start
def start():
    llm = OpenAI(temperature=0, streaming=True)
    llmc = ChatOpenAI(temperature=.7, streaming=True, model="gpt-3.5-turbo")
    search = GoogleSearchAPIWrapper()

    cl.user_session.set("llm", llm)
    cl.user_session.set("llmc", llmc)
    cl.user_session.set("search", search)
    memory = get_memory(llmc)
    tools = get_tools(llmc, memory)
    agent_chain = get_conversation_chain(llmc, memory, tools)

    cl.user_session.set("agent", agent_chain)
    cl.user_session.set("memory", memory)
    cl.user_session.set("tools", tools)

@cl.on_message
async def main(message: str):
    agent = cl.user_session.get("agent")
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)
    reply = None
    reply = await cl.make_async(agent.run)({"input": message}, callbacks=[cb])

    print(f"Reply: {reply}")
    await cl.Message(reply).send()
