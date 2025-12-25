from mcp.server.fastmcp import FastMCP
import json
import os

# Initialize FastMCP Server
mcp = FastMCP("riverline-profile-server")

PROFILE_FILE = "user_profile.json"

@mcp.tool()
def get_user_profile(user_id: str = "user_123") -> str:
    """
    Reads the full user profile including debt details and risk score.
    Use this to understand the user's background before negotiating.
    If no user_id is provided, it defaults to the current logged-in user.
    """
    if not os.path.exists(PROFILE_FILE):
        return "Error: Profile database not found."
    
    try:
        with open(PROFILE_FILE, "r") as f:
            data = json.load(f)
        
        # Look up the specific user
        user_data = data.get(user_id)
        if user_data:
            return json.dumps(user_data, indent=2)
        else:
            return f"Error: User {user_id} not found."
            
    except Exception as e:
        return f"Error reading profile: {str(e)}"

@mcp.tool()
def update_communication_preference(preference: str, user_id: str = "user_123") -> str:
    """
    Updates how the user wants to be contacted (e.g., 'WhatsApp', 'Email', 'Phone').
    """
    if not os.path.exists(PROFILE_FILE):
        return "Error: Profile database not found."
        
    try:
        with open(PROFILE_FILE, "r") as f:
            data = json.load(f)
        
        if user_id in data:
            data[user_id]["preferred_communication"] = preference
            
            with open(PROFILE_FILE, "w") as f:
                json.dump(data, f, indent=2)
            return f"Success: Updated preference to {preference} for {user_id}."
        else:
            return f"Error: User {user_id} not found."

    except Exception as e:
        return f"Error updating profile: {str(e)}"

if __name__ == "__main__":
    # fastmcp.run() handles stdio by default
    mcp.run()
