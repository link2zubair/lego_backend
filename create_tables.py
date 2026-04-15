"""Create all PostgreSQL tables for LEGO Vision."""
import asyncio, warnings
warnings.filterwarnings("ignore")

from database import engine, Base
import models  # registers User, ScanHistory, SavedBuild with Base

async def main():
    print("Connecting to PostgreSQL...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("\nSUCCESS! Tables created:")
    print("   -> users")
    print("   -> scan_history")
    print("   -> saved_builds")
    await engine.dispose()

asyncio.run(main())
