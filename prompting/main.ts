import { ChatCompletionFunctions, ChatCompletionResponseMessage, Configuration, OpenAIApi } from "openai";

const { OPENAI_KEY }  = process.env;
if (!OPENAI_KEY) {
    throw new Error("OPENAI_KEY env var must be set.");
}

const configuration = new Configuration({
  apiKey: OPENAI_KEY,
});
const openai = new OpenAIApi(configuration);

enum Function {
    CREATE_PLAN = "create_plan"
}

type LLMFunction<T extends {[param: string]: any}> = {
    funcdef: ChatCompletionFunctions
    helperFunctions: ChatCompletionFunctions[],
    promptGen: (params: T) => string
}

const createPlan: LLMFunction<{objective: string}> = {
    funcdef: {
        "name": "create_plan",
        "description": "Given an objective, create a plan for the objective.",
        "parameters": {
            "type": "object",
            "properties": {
                // "location": {
                //     "type": "string",
                //     "description": "The city and state, e.g. San Francisco, CA",
                // },
                // "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                "objective": {
                    "type": "string"
                }
            },
            "required": ["objective"],
        }
    },
    helperFunctions: [
        {
            name: "__data_plan",
            description: "A plan to fulfill an objective.",
            parameters: {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    }
                },
                "required": ["steps"],
            }
        },
        {
            name: "__data_atomic",
            description: "An objective that is too small to further plan.",
            parameters: {
                type: "object",
                properties: {}
            }
        }
    ],
    promptGen: ({objective}) => `Create a plan to fulfill the following objective: ${objective}. Invoke the function '__data_plan' with the steps of the plan.
Some objectives are too small in scope already to further break them down into steps. In that case, invoke the function named '__data_atomic'.`
};

type Expr = {
    kind: 'prompt',
    val: string
} | {
    kind: 'apply',
    funcname: string,
    params: {[param: string]: any}
} | {
    kind: 'response',
    val: {[key: string]: any} | string
}

async function complete(prompt: string, functions: ChatCompletionFunctions[]): Promise<ChatCompletionResponseMessage> {
    const chatCompletion = await openai.createChatCompletion({
        model: "gpt-3.5-turbo-0613",
        functions,
        messages: [{role: "user", content: prompt}],
    });
    return chatCompletion.data.choices[0].message!;
}

const llmFunctions: LLMFunction<any>[] = [createPlan];
const functions = llmFunctions.map(llmF => llmF.funcdef);

const DATA_FUNC_PREFIX = "__data_";
function isDataFunc(funcname: string) {
    return funcname.startsWith(DATA_FUNC_PREFIX);
}

async function deserializeCompletion(completion: ChatCompletionResponseMessage, context: LLMFunction<any>[]): Promise<Expr> {
    if (completion.function_call) {
        const functionCall = completion.function_call;

        const funcname = functionCall.name!;
        // TODO handle parse failures
        const funcargs = JSON.parse(functionCall.arguments!);

        if (isDataFunc(funcname)) {
            return {
                kind: 'response',
                val: {
                    __meta: 'data',
                    name: funcname,
                    data: funcargs
                }
            };
        } else {
            return evalExpr({
                kind: 'apply',
                funcname,
                params: funcargs
            }, context);
        }
    } else if (completion.content) {
        return {
            kind: 'response',
            val: completion.content
        }
    } else {
        throw new Error(`Fatal error: do not know how to handle completion. ${JSON.stringify(completion)}`)
    }
}

async function evalExpr(expr: Expr, context: LLMFunction<any>[]): Promise<Expr> {
    console.log(`eval: ${JSON.stringify(expr)}`);
    switch (expr.kind) {
        case 'prompt': {
            const completion = await complete(expr.val, context.map(f => f.funcdef));
            return deserializeCompletion(completion, context);
        }
        case 'apply': {
            const func = context.find(f => f.funcdef.name === expr.funcname)!;
            const completion = await complete(
                func.promptGen(expr.params),
                func.helperFunctions
            )
            return deserializeCompletion(completion, context);
        }
        case 'response':
            return expr;
    }
}

async function makePlan(superobjectives: string[], objective: string): Promise<string[]> {
    const superobjectivePrompts = superobjectives.map((o, i) =>
`${i+1}. ${o}`);

    let prompt: string;
    if (superobjectives.length === 0) {
        prompt = `I need help creating a plan for the following objective: ${objective}`;
    } else {
        const superobjectivePrompt = superobjectivePrompts.join("\n");
        prompt = `I am in the process of completing the following objectives. They're ordered in a list, and each objective is a part of the plan to complete the immediately previous objective in the list.:
${superobjectivePrompt}
        
Help me create a plan for the objective I'm currently focusing on: ${objective}`;
    }

    const resp = await evalExpr({
        kind: "prompt",
        val: prompt
    }, llmFunctions);

    if (resp.kind === 'response') {
        const respdata = resp.val as Record<string, any>;

        if (!respdata.__meta || respdata.__meta !== 'data') {
            throw new Error(`Fatal error: unexpected response ${JSON.stringify(resp)}`);
        }

        switch (respdata.name) {
            case '__data_plan':
                const { steps: rawSteps } = (resp.val as Record<string, any>)?.data;
                const steps = rawSteps as string[];
                const indent = superobjectives.map(_ => ">").join("");
                console.log(
                    steps.map(step => `${indent} ${step}`)
                );

                for (const step of steps) {
                    const newSuperObjectives = [
                        ...superobjectives,
                        objective
                    ];
                    const stepPlan = await makePlan(newSuperObjectives, step);
                    steps.push(...stepPlan);
                }

                return steps;
            case '__data_atomic':
                return [resp.val!] as string[];
            default:
                throw new Error(`Fatal error: unknown data name ${respdata.name}`)
        }
    } else {
        throw new Error(`Unexpected response: ${JSON.stringify(resp)}`);
    }
}

async function main() {
    const resp = await makePlan([], "plan a wedding");
    console.log(resp);
}

main();