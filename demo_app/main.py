"""Python file to serve as the frontend"""
from langchain import OpenAI
from langchain.agents import Tool, AgentExecutor, OpenAIMultiFunctionsAgent
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.tools import DuckDuckGoSearchRun
from langchain.schema.messages import SystemMessage, HumanMessage
from langchain.callbacks.base import BaseCallbackHandler
import random
import chainlit as cl


def create_chain(chain_type, llm, input_variables, template_str, memory):
    """Create a chain of agents"""

    if template_str:
        prompt = PromptTemplate(
            input_variables=input_variables,
            template=template_str,
        )

    if chain_type == LLMMathChain:
        return chain_type(
            llm=llm,
            prompt=prompt,
            verbose=True,
            memory=memory,
            input_key="input",
        )

    return chain_type(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=ReadOnlySharedMemory(memory=memory),
)

def get_memory(llm):
    """Create a memory buffer for the conversation"""

    memory_buffer = ConversationSummaryBufferMemory(
        llm=llm,
        memory_key="chat_history",
        max_token_limit=12000,
        ai_prefix="Game Master",
        human_prefix="Protagonist",
        input_key="input",
        return_messages=True,
    )

    return memory_buffer

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
    )

    agent_chain = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory,
        handle_parsing_errors="As a Storyteller, that doesn't quite make sense.  What else can I try?",
        max_iterations=3,
    )

    return agent_chain

def get_llmmath_chain(llm):
    """Create a chain of agents for the LLMMath tool"""

    agent = LLMMathChain.from_llm(
        llm=llm,
        verbose=True,
    )

    return agent

def roll_dice(x: int, y: int) -> int:
    """Roll x dice with y sides each"""

    return sum([random.randint(1, y) for _ in range(x)])

def roll_dice_parser(input_str: str) -> str:
    """Parse a string for dice rolls"""

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

def get_tools(llm, memory):
    """Create a list of tools"""

    tools_data = {
        "Dice_Rolling_Assistant": {
            "description": "An assistant that can roll any combination of dice. Useful for adding unpredictability and uniqueness to story elements. Dice roll results represent the positive (high) or negative (low) outcomes of events against a predetermined difficulty value. The input to this tool should follow the exact format XdY where X is the number of dice, and Y is the number of sides on each die rolled (e.g. 4d20, or 1d6).",
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
        elif 'coroutine' in tool_info:
            coroutine = tool_info['coroutine']
            tool = Tool(name=tool_name, func=None, coroutine=coroutine, description=description)
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
    """Start the conversation"""

    llm = OpenAI(temperature=0, streaming=True)
    llmc = ChatOpenAI(temperature=.7, streaming=True, model="gpt-3.5-turbo-16k")
    search = DuckDuckGoSearchRun()

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
        if not token:
            return
        await self.msg.stream_token(token)

    async def on_llm_end(self, response: str, **kwargs):
        await self.msg.send()
        self.msg = cl.Message(content="")

@cl.on_message
async def main(message: str):
    """Main function"""

    agent = cl.user_session.get("agent")
    cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True)
    #import ipdb; ipdb.set_trace()
    reply = await agent.acall(
        {
            "input": message,
        },
        callbacks=[
            cl.AsyncLangchainCallbackHandler(stream_final_answer=True),
            StreamHandler()
        ]
    )
    print(f"Reply: {reply}")
    
    agent.memory.prune()

