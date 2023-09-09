if __name__ == "__main__":
    import uvicorn

    uvicorn.run("modules.api_chatbot:app",
                host="0.0.0.0", reload=True, port=443,
                ssl_keyfile='/etc/nginx/ssl/ai-town.mcglobal.ai.key',
                ssl_certfile='/etc/nginx/ssl/ai-town.mcglobal.ai.pem')
