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
        memory=ReadOnlySharedMemory(memory=memory),
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
        ai_prefix="Dungeon Master",
        human_prefix="Player Character",
        input_key="input",
    )
    memory_entity = ConversationKGMemory(
        llm=llm,
        memory_key="entities",
        input_key="input",
    )

    combined = CombinedMemory(memories=[memory_buffer, memory_entity])

    return combined

def get_conversation_chain(llm, memory, tools):

    prompt = PromptTemplate(
        input_variables=["input", "buffer", "entities"],
        template="""As the Dungeon Master for this D&D 5e campaign, you should decide what to say next to the player.

What should I say to the player?:
{input}

Campaign History:
{buffer}

Campaign Entities Data:

{entities}
"""
    )

    prefix = """Reply to the player's message directly as this one person one shot campaign's Dungeon Master...

You have access to the following tools:
    """
    suffix = """You are a Dungeon Master replying to a player's message in a one shot D&D 5e campaign. Only reply as a real Dungeon Master would, and don't ever break character.

Player's message:
{input}

Campaign History:
{buffer}

Campaign Entities Data:
{entities}
"""
#{agent_scratchpad}
#"""
    # Dungeon Master Scratchpad: {agent_scratchpad}

    prompt = ZeroShotAgent.create_prompt(tools, prefix=prefix, suffix=suffix, input_variables=["input", "buffer", "entities"])
    llm_chain = ConversationChain(llm=llm, prompt=prompt, verbose=True, memory=memory)

    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory,
        handle_parsing_errors="As a Dungeon Master, I'll put it a different way...",
        max_iterations=10,
        early_stopping_method="generate",
    )

    #agent = initialize_agent(
    #    llm=llm,
    #    agent_type=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    #    tools=tools,
    #    memory=memory,
    #    verbose=True,
    #    max_iterations=10,
    #    early_stopping_method="generate",
    #    handle_parsing_errors="As a Dungeon Master, I'll put it a different way...",
    #    prompt=prompt,
    #    ai_prefix="Dungeon Master",
    #    human_prefix="Player Character",
    #    input_key="input",
    #)

    #return agent

    return agent_chain

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
        ai_prefix="Dungeon Master",
        human_prefix="Player Character",
    )

    return agent

def get_tools(llm, memory):
    tools_data = {
        "Dungeon Master Speaks": {
            "template": """As the Dungeon Master for this D&D 5e one shot campaign, I should decide what to say next to the player. Only reply as a real Dungeon Master would, and don't ever break character. 

What should I say to the player?:
{input}

Campaign History:
{buffer}

Campaign Entities Data:
{entities}

""",
            "description": "Useful for when the Dungeon Master is done considering what the player said and looking up any information they need. The input to this tool should be what the player said, followed by the Dungeon Master's current notes.  The result of this tool should be the Dungeon Master's response to the player.",
            "input_variables": ["input", "buffer", "entities"],
            "chain_type": ConversationChain,
            "return_direct": True,
        },
        "Dungeon Master Considering": {
            "template": """As the Dungeon Master for this D&D 5e campaign, you should consider what should happen next.

I want to consider the following:
{input}

Campaign History:
{buffer}

Campaign Entities Data:
{entities}

""",
            "description": "Useful for when the Dungeon Master is unsure of what to do next and needs to consider their options. The input to this tool should be what the Dungeon Master is considering based on what the player said.  The result of this tool should be the Dungeon Master's thoughts on the topic.",
            "input_variables": ["input", "buffer", "entities"],
            "chain_type": ConversationChain,
        },
        "Campaign Start": {
            "template": """A new D&D is about to begin. As the Dungeon Master for this D&D 5e campaign, you should get the ball rolling. Only reply as a real Dungeon Master would, and don't ever break character.

What should I say to begin this campaign?:
{input}

Campaign History:
{buffer}

Campaign Entities Data:
{entities}

""",
            "description": "Initiates the start of a new D&D campaign, setting the scene and introducing initial plot elements. Useful for when a new campaign or session is starting. The input to this tool should be a unique one sentence made up story description based on what the player just said. The output of this tool should be what the Dungeon Master says to start the campaign.",
            "input_variables": ["input", "buffer", "entities"],
            "chain_type": ConversationChain,
            "return_direct": True,
        },
        "Dice Rolling Assistant": {
            "description": "An assistant that can roll any combination of dice. The input to this tool should be a comma separated list of numbers, each pair representing the number of dice followed by the number or sides on each die. For example, '1, 6, 10, 4' would be the input if you wanted to roll 1d6 and 10d4 dice. Make sure the rolls follow Dungeons and Dragons 5e rules.",
            "function": roll_dice_parser,
        },
        "Dungeons and Dragons Reference": {
            "description": "Searches the internet for information about Dungeons and Dragons.",
            "function": cl.user_session.get("search").run,
        },
    }
    
    # Remove "Dungeon Master Considering" tool for now
    del tools_data["Dungeon Master Considering"]
    del tools_data["Campaign Start"]
    #del tools_data["Dungeon Master Speaks"]

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
        reply = await cl.make_async(agent.run)({"input": message}, callbacks=[cb])
        #reply = await cl.make_async(agent.run)(message, callbacks=[cb])
    elif cl.user_session.get("chain_type") == ConversationChain:
        reply = await cl.make_async(agent.run)({"input": message}, callbacks=[cb])
    elif cl.user_session.get("chain_type") == LLMChain:
        reply = await cl.make_async(agent.run)({"input": message}, callbacks=[cb])
    print(f"Reply: {reply}")

    await cl.Message(reply).send()
