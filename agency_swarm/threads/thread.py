import inspect
import time
from typing import Union

from agency_swarm.agents import Agent
from agency_swarm.messages import MessageOutput
from agency_swarm.user import User
from agency_swarm.util.oai import get_openai_client


class Thread:
    id = None
    api_id: str = None
    thread = None
    run = None

    def __init__(self, agent: Union[Agent, User], recipient_agent: Agent):
        self.agent = agent
        self.recipient_agent = recipient_agent

    @classmethod
    def from_model(cls, thread_model):
        thread = cls.__new__(cls)
        thread.id = thread_model.id
        thread.api_id = thread_model.api_id
        thread.agent = User() if thread_model.agent is None else Agent.from_model(
            thread_model.agent)
        thread.recipient_agent = Agent.from_model(thread_model.recipient_agent)
        return thread

    def get_completion(self, message: str, yield_messages=True):
        client = get_openai_client()
        if not self.thread:
            if self.api_id:
                self.thread = client.beta.threads.retrieve(self.api_id)
            else:
                self.thread = client.beta.threads.create()
                self.api_id = self.thread.id

        # Check if a run is active
        if self.run and self.run.status in ['queued', 'in_progress']:
            while self.run.status in ['queued', 'in_progress']:
                print("Detected active run - sleeping")
                time.sleep(0.5)
                self.run = client.beta.threads.runs.retrieve(
                    thread_id=self.thread.id,
                    run_id=self.run.id
                )

        # send message
        client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=message
        )

        if yield_messages:
            yield MessageOutput("text", self.agent.name, self.recipient_agent.name, message)

        # create run
        self.run = client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.recipient_agent.id,
        )

        while True:
            # wait until run completes
            while self.run.status in ['queued', 'in_progress']:
                time.sleep(0.5)
                self.run = client.beta.threads.runs.retrieve(
                    thread_id=self.thread.id,
                    run_id=self.run.id
                )

            # function execution
            if self.run.status == "requires_action":
                tool_calls = self.run.required_action.submit_tool_outputs.tool_calls
                tool_outputs = []
                for tool_call in tool_calls:
                    if yield_messages:
                        yield MessageOutput("function", self.recipient_agent.name, self.agent.name, str(tool_call.function))

                    output = self._execute_tool(tool_call)
                    if inspect.isgenerator(output):
                        try:
                            while True:
                                item = next(output)
                                if isinstance(item, MessageOutput) and yield_messages:
                                    yield item
                        except StopIteration as e:
                            output = e.value
                    else:
                        if yield_messages:
                            yield MessageOutput("function_output", tool_call.function.name, self.recipient_agent.name, output)

                    tool_outputs.append(
                        {"tool_call_id": tool_call.id, "output": str(output)})

                # submit tool outputs
                self.run = client.beta.threads.runs.submit_tool_outputs(
                    thread_id=self.thread.id,
                    run_id=self.run.id,
                    tool_outputs=tool_outputs
                )
            # error
            elif self.run.status == "failed":
                raise Exception("Run Failed. Error: ", self.run.last_error)
            # return assistant message
            else:
                messages = client.beta.threads.messages.list(
                    thread_id=self.api_id
                )
                message = messages.data[0].content[0].text.value

                if yield_messages:
                    yield MessageOutput("text", self.recipient_agent.name, self.agent.name, message)

                return message

    def _execute_tool(self, tool_call):
        tools = self.recipient_agent.tools
        tool = next((tool for tool in tools if tool.__class__.__name__ ==
                    tool_call.function.name), None)

        if not tool:
            return f"Error: Tool {tool_call.function.name} not found. Available tools: {[tool.__class__.__name__ for tool in tools]}"

        try:
            # Zero out all non-private fields
            for attr in tool.__dict__:
                if not attr.startswith('_'):
                    setattr(tool, attr, None)

            # Set fields on the tool instance
            arguments = eval(tool_call.function.arguments)
            for attr, value in arguments.items():
                setattr(tool, attr, value)

            # Validate the tool model
            tool = type(tool).model_validate(tool)

            # get outputs from the tool
            output = tool.run()

            return output
        except Exception as e:
            return "Error: " + str(e)
