import { ChatCompletionFunctions, ChatCompletionRequestMessage, ChatCompletionResponseMessage, Configuration, CreateChatCompletionResponseChoicesInner, OpenAIApi } from "openai";
import * as readline from 'node:readline/promises';  // This uses the promise-based APIs
import { stdin as input, stdout as output } from 'node:process';

const { OPENAI_KEY }  = process.env;
if (!OPENAI_KEY) {
    throw new Error("OPENAI_KEY env var must be set.");
}

const configuration = new Configuration({
  apiKey: OPENAI_KEY,
});
const openai = new OpenAIApi(configuration);

interface Tool<T extends {[param: string]: any}> {
    funcdef: ChatCompletionFunctions,
    name: string;
    use: (params: T) => Promise<string>
};

class GoogleTool implements Tool<{query: string}> {
    funcdef = {
        name: 'google',
        description: "Query the Google search engine",
        parameters: {
            type: "object",
            properties: {
                query: {
                    type: "string"
                }
            },
            required: ["query"]
        }
    };

    get name(): string {
        return 'google';
    }
    
    async use(_: {query: string}) {
        // return `This is the Google result for the following query: ${query}`;
        return `Lebron James has played for the Cavaliers, Heat, and the Lakers.`;
    }
}

async function complete(messages: ChatCompletionRequestMessage[], functions: ChatCompletionFunctions[]): Promise<{message: ChatCompletionResponseMessage, halt: boolean}> {
    const chatCompletion = await openai.createChatCompletion({
        model: "gpt-3.5-turbo-0613",
        functions,
        messages
    });

    const choice: CreateChatCompletionResponseChoicesInner = chatCompletion.data.choices[0];
    return {
        message: choice.message!,
        halt: choice.finish_reason === 'stop'
    };
}

const BASE_PROMPT: ChatCompletionRequestMessage = {
    role: "system",
    content: `You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are given in the 'functions' parameter.

Example session:

Question: What is the capital of France?
Thought: I should look up France on Google
Action: google: France
PAUSE

You will be called again with this:

Observation: France is a country. The capital is Paris.

You then output:

Answer: The capital of France is Paris`
};

const rl = readline.createInterface({ input, output });

async function loop() {
    const tools: Tool<any>[] = [
        new GoogleTool()
    ]
    const funcdefs = tools.map(t => t.funcdef);
    const messages: ChatCompletionRequestMessage[] = [BASE_PROMPT];

    const answer = await rl.question('User: ');
    const newMessage: ChatCompletionRequestMessage = {
        role: "user",
        content: answer
    }
    messages.push(newMessage);

    while (true) {
        const {message: completion, halt} = await complete(messages, funcdefs)

        messages.push(completion);
        if (completion.content) {
            console.log(`Assistant: ${completion.content}`)
        }

        if (halt) {
            break;
        }

        if (completion.function_call) {
            const { name: funcName, arguments: rawFuncArgs } = completion.function_call;

            console.log(`[${funcName}]: rawFuncArgs`)
            const tool = tools.find(t => t.name === funcName);
            if (!tool) {
                throw new Error(`Cannot find function ${funcName}`);
            }

            const funcArgs = JSON.parse(rawFuncArgs!);
            const toolResult = await tool.use(funcArgs);
            console.log(`Observation: ${toolResult}`);
            messages.push({
                role: 'function',
                name: funcName,
                content: toolResult
            })
        }
    }

    console.log("End session.");
}

loop();