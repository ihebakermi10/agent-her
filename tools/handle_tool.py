from google import genai
from tool import tweet_posting_tool, generate_image_handler

async def handle_tool_call(session, tool_call):

    for fc in tool_call.function_calls:
        fn_name = fc.name
        fn_id = fc.id
        args = fc.args or {}

        if fn_name == "tweet_posting_tool":
            tweet_str = args.get("tweet", "")
            result_msg = tweet_posting_tool(tweet_str)
            tool_response = genai.types.LiveClientToolResponse(
                function_responses=[
                    genai.types.FunctionResponse(
                        name=fn_name,
                        id=fn_id,
                        response={"result": result_msg}
                    )
                ]
            )
            print(f"\n>>> tweet_posting_tool appelé avec tweet='{tweet_str}', retourne : {result_msg}")
            await session.send(tool_response)

        elif fn_name == "generate_image":
            prompt = args.get("prompt", "")
            result = await generate_image_handler(prompt)
            if "error" in result:
                result_msg = result["error"]
            else:
                result_msg = f"Image générée et sauvegardée à : {result['image_path']}"
            tool_response = genai.types.LiveClientToolResponse(
                function_responses=[
                    genai.types.FunctionResponse(
                        name=fn_name,
                        id=fn_id,
                        response={"result": result_msg}
                    )
                ]
            )
            print(f"\n>>> generate_image appelé avec prompt='{prompt}', retourne : {result_msg}")
            await session.send(tool_response)

        else:
            error_resp = genai.types.LiveClientToolResponse(
                function_responses=[
                    genai.types.FunctionResponse(
                        name=fn_name,
                        id=fn_id,
                        response={"result": f"Fonction inconnue {fn_name}"}
                    )
                ]
            )
            await session.send(error_resp)
