import asyncio
from sqlalchemy import text
import database as db_module

async def check():
    async with db_module.engine.begin() as conn:
        result = await conn.execute(text(
            "SELECT column_name, data_type, character_maximum_length "
            "FROM information_schema.columns "
            "WHERE table_name = 'users' AND column_name = 'avatar_url'"
        ))
        row = result.fetchone()
        print(f"Column: {row[0]}  |  Type: {row[1]}  |  Max length: {row[2]}")

asyncio.run(check())
