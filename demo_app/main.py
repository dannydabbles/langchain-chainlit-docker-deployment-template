"""Python file to serve as the frontend"""
from langchain import OpenAI # LLMMathChain SerpAPIWrapper?
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import initialize_agent, ZeroShotAgent, Tool, AgentExecutor, AgentType, OpenAIMultiFunctionsAgent
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationTokenBufferMemory, ConversationKGMemory, CombinedMemory, ReadOnlySharedMemory
from langchain.chains import LLMMathChain, LLMChain, ConversationChain
from langchain.schema.messages import SystemMessage, HumanMessage
from langchain.callbacks.base import BaseCallbackHandler
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
    #import ipdb; ipdb.set_trace()

    words = input_str.split()
    dice = []
    for word in words:
        # If word does not start with a number
        if word[0].isdigit():
            dice.append(word)

    # Roll all dice
    results = []
    for die in dice:
        x, y = die.split("d")
        results.append(roll_dice(int(x), int(y)))

    return sum(results)

def get_memory(llm):
    memory_buffer = ConversationTokenBufferMemory(
        llm=llm,
        memory_key="chat_history",
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

    return CombinedMemory(memories=[memory_buffer, memory_entity])

def get_conversation_chain(llm, memory, tools):
    #import ipdb; ipdb.set_trace()
    prefix = """Decide what to do next as Game Mater of this story, based on the following current story information:

Protagonist's Message (Game Master's Notes):
{input}

Story History (Game Master's Notes):
{chat_history}

Story Entities (Game Master's Notes):
{entities}

Game Master's Scratchpad (Game Master's Notes):
{agent_scratchpad}

You have access to the following tools:"""
    format_instructions = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer\nFinal Answer: the final answer to the original input question"""
    suffix = """Begin!

Question: {input}
Thought:{agent_scratchpad}"""

    #prompt = ZeroShotAgent.create_prompt(tools, prefix=prefix, suffix=suffix, format_instructions=format_instructions, input_variables=["input", "chat_history", "entities", "agent_scratchpad"])

    #llm_chain = LLMChain(
    #        llm=llm,
    #        prompt=prompt,
    #        verbose=True,
    #        memory=ReadOnlySharedMemory(memory=memory),
    #)

    #agent = ZeroShotAgent(
    #    llm_chain=llm_chain,
    #    tools=tools,
    #    verbose=True,
    #    ai_prefix="Game Master",
    #    human_prefix="Protagonist",
    #)

    #prompt = OpenAIAgent.create_prompt(
    #    tools,
    #    #prefix=prefix,
    #    #suffix=suffix,
    #    #format_instructions=format_instructions,
    #    input_variables=["input", "chat_history", "entities", "agent_scratchpad"]
    #)

    #import ipdb; ipdb.set_trace()

    prompt = OpenAIMultiFunctionsAgent.create_prompt(
        system_message=SystemMessage(content="You are an AI Game Master for a D&D 5e campaign, what do you say next?")
    )

    #import ipdb; ipdb.set_trace()

    agent = OpenAIMultiFunctionsAgent(
        llm=llm,
        prompt=prompt,
        tools=tools,
        verbose=True,
    )

    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory,
        handle_parsing_errors="As a Storyteller, that doesn't quite make sense.  What else can I try?",
        max_iterations=3,
        early_stopping_method="generate",
        prompt=prompt,
    )

    #import ipdb; ipdb.set_trace()

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
        "Dice_Rolling_Assistant": {
            "description": "An assistant that can roll any combination of dice. Useful for adding unpredictability and uniqueness to story elements. Dice roll results represent the positive (high) or negative (low) outcomes of events against a predetermined difficulty value. The input to this tool should follow the format XdY where X is the number of dice, and Y is the number of sides on each die rolled.",
            "function": roll_dice_parser,
        },
        "Internet_Search": {
            "description": "Useful for looking up unknown information about D&D 5e rules, lore, and other fine details. The input to this tool should be a google search query. Ask targeted questions! The result of this tool should be a summary of the search results.",
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


class StreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.msg = cl.Message(content="")

    async def on_llm_new_token(self, token: str, **kwargs):
        await self.msg.stream_token(token)

    async def on_llm_end(self, response: str, **kwargs):
        await self.msg.send()
        self.msg = cl.Message(content="")

@cl.on_message
async def main(message: str):
    agent = cl.user_session.get("agent")
    cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True)
    await agent.arun(
        {
            "input": message
        },
        callbacks=[
            cl.AsyncLangchainCallbackHandler(stream_final_answer=True),
            StreamHandler()
        ]
    )

