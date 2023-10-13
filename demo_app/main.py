"""Python file to serve as the frontend"""
from langchain import OpenAI # LLMMathChain SerpAPIWrapper?
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import initialize_agent, ZeroShotAgent, Tool, AgentExecutor, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationTokenBufferMemory, ConversationKGMemory, CombinedMemory
from langchain.chains import LLMMathChain, LLMChain, ConversationChain
import os
import chainlit as cl
from langchain import hub


def create_chain(chain_type, llm, input_variables, template_str, memory):
    prompt = PromptTemplate(
        input_variables=input_variables,
        template=template_str,
    )

    return chain_type(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory,
    )

def roll_dice(x: int, y: int) -> int:
    return sum([random.randint(1, y) for _ in range(x)])

def get_memory(llm):
    memory_buffer = ConversationTokenBufferMemory(
        llm=llm,
        memory_key="buffer",
        max_tokens=1000,
        ai_prefix="Dungeon Master:",
        human_prefix="Player Character:",
        input_key="input",
    )
    memory_entity = ConversationKGMemory(
        llm=llm,
        memory_key="entities",
        input_key="input",
    )

    combined = CombinedMemory(memories=[memory_buffer, memory_entity])

    return combined

def get_conversation_chain(llm, memory, tools, prompt="You are an AI Dungeon Master for a D&D 5e campaign, what do you say next?"):

    agent = initialize_agent(
        llm=llm,
        agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        tools=tools,
        memory=memory,
        verbose=True,
        max_iterations=3,
        early_stopping_method="generate",
        handle_parsing_errors="As a Dungeon Master, I'll put it a different way...",
        prompt=prompt,
        ai_prefix="Dungeon Master:",
        human_prefix="Player Character:",
    )

    return agent

def get_llmmath_chain(llm, memory, tools, prompt="You are an AI Dungeon Master for a D&D 5e campaign, what do you say next?"):

    agent = LLMMathChain(
        llm=llm,
        tools=tools,
        memory=memory,
        verbose=True,
        max_iterations=10,
        early_stopping_method="generate",
        handle_parsing_errors="As a Dungeon Master, I'll put it a different way...",
        prompt=prompt,
        ai_prefix="Dungeon Master:",
        human_prefix="Player Character:",
    )

    return agent

def get_tools(llm, memory):
    tools_data = {
        "Dungeon Master Speaks": {
            "template": """As the Dungeon Master for this D&D 5e campaign, I should decide what to say next if I'm done considering what the player said and looking up any information I need.

What should happen next after a player says the following?:
{input}

Campaign History:
{buffer}

Campaign Entities Data:

{entities}

Dungeon Master:""",
            "description": "Useful for when the Dungeon Master is done considering what the player said and looking up any information they need.",
            "input_variables": ["input", "buffer", "entities"],
            "chain_type": ConversationChain,
            "return_direct": True,
        },
        "Dungeon Master Considering": {
            "template": """As the Dungeon Master for this D&D 5e campaign, you should consider what should happen next.

What should happen next after a player says the following?:
{input}

Campaign History:
{buffer}

Campaign Entities Data:

{entities}

Dungeon Master:""",
            "description": "Useful for when the Dungeon Master is unsure of what to do next and needs to consider their options.",
            "input_variables": ["input", "buffer", "entities"],
            "chain_type": ConversationChain,
        },
        "Campaign Start": {
            "template": """A new D&D is about to begin. As the Dungeon Master for this D&D 5e campaign, you should get the ball rolling.

What should happen next after a player says the following?:
{input}

Campaign History:
{buffer}

Campaign Entities Data:

{entities}

Dungeon Master:""",
            "description": "Initiates the start of a new D&D campaign, setting the scene and introducing initial plot elements. Useful for when a new campaign or session is starting.",
            "input_variables": ["input", "buffer", "entities"],
            "chain_type": ConversationChain,
        },
        "Dice Check": {
            "template": """As Dungeon Master for the current D&D 5e campaign, you should roll dice to determine what happens next.  You may choose to reveal the DC for the roll or not.

What roll should happen next after a player says the following?:
{input}

Campaign History:
{buffer}

Campaign Entities Data:
{entities}

Dungeon Master:""",
            "description": "Determines whether a dice check is needed in response to a player's action, and what the parameters of that check should be. Useful for integrating D&D 5e gameplay mechanics into the narrative.",
            "input_variables": ["input", "buffer", "entities"],
            "chain_type": LLMMathChain,
        },
        "Dungeons and Dragons Reference": {
            "description": "Searches the internet for information about Dungeons and Dragons.",
            "function": cl.user_session.get("search").run,
        },
    }

    templates = []
    descriptions = []
    tools = []
    for tool_name, tool_info in tools_data.items():
        description = tool_info["description"]

        if 'function' in tool_info:
            func = tool_info['function']
            tool = Tool(name=tool_name, func=func, description=description)
        else:
            template_str = tool_info["template"]
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
    llmc = ChatOpenAI(temperature=0, streaming=True, model="gpt-3.5-turbo")
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
    if cl.user_session.get("chain_type") == LLMMathChain:
        reply = await cl.make_async(agent.run)(message, callbacks=[cb])
    elif cl.user_session.get("chain_type") == ConversationChain:
        reply = await cl.make_async(agent.run)({"input": message}, callbacks=[cb])
    print(f"Reply: {reply}")
