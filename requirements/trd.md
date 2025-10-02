# Technical Requirements Document (TRD) Template

LEGEND: {{ }} denotes that the sections should be filled out with the specific agent information.

## 1. Executive Summary
### 1.1 Purpose
This document outlines the technical requirements for implementing a {{ AGENT TYPE }} agent using Pydantic AI that maintains conversation history across multiple interactions. The agent will provide {{ RESPONSE TYPE }} while preserving context from previous queries.

### 1.2 Scope
The Agent will:
- {{ AGENT SCOPE LIST }}

_Use action verbs; exclude non-goals._


## 2. System Overview
### 2.1 High-Level Architecture
```plaintext
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│ run_agent() │────▶│  Pydantic AI │────▶│     LLM      │
│             │◀────│    Agent     │◀────│    Model     │
└─────────────┘     └──────────────┘     └──────────────┘
                            │
                            │
                    ┌───────▼────────┐
                    │   MCP Server   │
                    └───────┬────────┘
                ┌───────────┴──────────┐
                │                      │
        ┌───────▼────────┐     ┌───────▼────────┐
        │  {{ Tool }}    │     │  {{ Tool }}    │
        └───────┬────────┘     └───────┬────────┘
                │                      │
        ┌───────▼────────┐     ┌───────▼────────┐
        │      API       │     │      API       │
        └────────────────┘     └────────────────┘
```
### 2.2 Technology Stack
Framework: Pydantic AI Framework
Language Model: claude-sonnet-4-20250514 with gpt-5-mini as fallback
MCP Server: MCPServerStreamableHTTP from Pydantic AI
Data Validation: Pydantic BaseModel
Async Runtime: Python asyncio
Logging/Monitoring: Logfire (optional)
Python Version: 3.11+

## 3. Functional Requirements
### 3.1 Core Features
#### 3.1.1 {{ CORE FEATURE 1}}
{{ MORE CORE FEATURES}}

#### 3.1.x Conversation History Management
FR-3.1.1: The agent SHALL maintain a complete message history across conversation turns using Pydantic AI message history
FR-3.1.2: The agent SHALL preserve context from previous queries
FR-3.1.3: The agent SHALL support follow-up questions about previous responses

### 3.2 Tool Functionality
FR-3.2.1: The agent SHALL use MCP server tools when [condition/intent criteria], otherwise respond directly.

## 4. Non-functional Requirements
### 4.1 Observability with Logfire
...

## 5. Data Requirements
### 5.1 Data Models
**Never create duplicate models.** Always check Pydantic AI API reference first. e.g. If we need to define usage limits, Pydantic AI already has UsageLimits data model.

#### 5.1.1 {{ DATA MODEL 1}}
{{ MORE DATA MODELS}}

### 5.2 Message History Structure
**From Pydantic AI**
- ModelRequest: User prompts and system instructions
- ModelResponse: LLM responses and tool calls
- ToolReturnPart: Results from tool executions
- RetryPromptPart: Error recovery messages

## 6. Interface Specifications
### 6.1 Agent Configuration

```python
from pydantic_ai import Agent

def create_agent() -> Agent:
    agent = Agent(
        model="[model name]",
        instructions="[system instructions]",
        # deps_type=Deps,            # verify exact param name if supported
        # result_type=OutputSchema,  # optional: for structured outputs
        # retries=3,                 # verify param name
        # tools=[remote_mcp_server], # verify param name ('tools' vs 'toolsets')
        # Additional configuration
    )

    # Additional configuration
    return agent

agent = create_agent()
```

### 6.2 MCP Server Configuration

#### 6.2.1 Available MCP Tools
The following tools are available from the MCP server:

```json
[
  {
    "name": "hf_whoami",
    "description": "Hugging Face tools are being used anonymously and may be rate limited. Call this tool for instructions on joining and authenticating.",
    "inputSchema": {
      "type": "object",
      "properties": {},
      "additionalProperties": false,
      "$schema": "http://json-schema.org/draft-07/schema#"
    }
  },
  {
    "name": "space_search",
    "description": "Find Hugging Face Spaces using semantic search. Include links to the Space when presenting the results.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "minLength": 1,
          "maxLength": 100,
          "description": "Semantic Search Query"
        },
        "limit": {
          "type": "number",
          "default": 10,
          "description": "Number of results to return"
        },
        "mcp": {
          "type": "boolean",
          "default": false,
          "description": "Only return MCP Server enabled Spaces"
        }
      },
      "required": [
        "query"
      ],
      "additionalProperties": false,
      "$schema": "http://json-schema.org/draft-07/schema#"
    }
  },
  {
    "name": "model_search",
    "description": "Find Machine Learning models hosted on Hugging Face. Returns comprehensive information about matching models including downloads, likes, tags, and direct links. Include links to the models in your response",
    "inputSchema": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "Search term. Leave blank and specify \"sort\" and \"limit\" to get e.g. \"Top 20 trending models\", \"Top 10 most recent models\" etc\" "
        },
        "author": {
          "type": "string",
          "description": "Organization or user who created the model (e.g., 'google', 'meta-llama', 'microsoft')"
        },
        "task": {
          "type": "string",
          "description": "Model task type (e.g., 'text-generation', 'image-classification', 'translation')"
        },
        "library": {
          "type": "string",
          "description": "Framework the model uses (e.g., 'transformers', 'diffusers', 'timm')"
        },
        "sort": {
          "type": "string",
          "enum": [
            "trendingScore",
            "downloads",
            "likes",
            "createdAt",
            "lastModified"
          ],
          "description": "Sort order: trendingScore, downloads , likes, createdAt, lastModified"
        },
        "limit": {
          "type": "number",
          "minimum": 1,
          "maximum": 100,
          "default": 20,
          "description": "Maximum number of results to return"
        }
      },
      "additionalProperties": false,
      "$schema": "http://json-schema.org/draft-07/schema#"
    }
  },
  {
    "name": "paper_search",
    "description": "Find Machine Learning research papers on the Hugging Face hub. Include 'Link to paper' When presenting the results. Consider whether tabulating results matches user intent.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "minLength": 3,
          "maxLength": 200,
          "description": "Semantic Search query"
        },
        "results_limit": {
          "type": "number",
          "default": 12,
          "description": "Number of results to return"
        },
        "concise_only": {
          "type": "boolean",
          "default": false,
          "description": "Return a 2 sentence summary of the abstract. Use for broad search terms which may return a lot of results. Check with User if unsure."
        }
      },
      "required": [
        "query"
      ],
      "additionalProperties": false,
      "$schema": "http://json-schema.org/draft-07/schema#"
    }
  },
  {
    "name": "dataset_search",
    "description": "Find Datasets hosted on the Hugging Face hub. Returns comprehensive information about matching datasets including downloads, likes, tags, and direct links. Include links to the datasets in your response",
    "inputSchema": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "Search term. Leave blank and specify \"sort\" and \"limit\" to get e.g. \"Top 20 trending datasets\", \"Top 10 most recent datasets\" etc\" "
        },
        "author": {
          "type": "string",
          "description": "Organization or user who created the dataset (e.g., 'google', 'facebook', 'allenai')"
        },
        "tags": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "Tags to filter datasets (e.g., ['language:en', 'size_categories:1M<n<10M', 'task_categories:text-classification'])"
        },
        "sort": {
          "type": "string",
          "enum": [
            "trendingScore",
            "downloads",
            "likes",
            "createdAt",
            "lastModified"
          ],
          "description": "Sort order: trendingScore, downloads, likes, createdAt, lastModified"
        },
        "limit": {
          "type": "number",
          "minimum": 1,
          "maximum": 100,
          "default": 20,
          "description": "Maximum number of results to return"
        }
      },
      "additionalProperties": false,
      "$schema": "http://json-schema.org/draft-07/schema#"
    }
  },
  {
    "name": "hub_repo_details",
    "description": "Get details for one or more Hugging Face repos (model, dataset, or space). Auto-detects type unless specified.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "repo_ids": {
          "type": "array",
          "items": {
            "type": "string",
            "minLength": 1
          },
          "minItems": 1,
          "maxItems": 10,
          "description": "Repo IDs for (models|dataset/space) - usually in author/name format (e.g. openai/gpt-oss-120b)"
        },
        "repo_type": {
          "type": "string",
          "enum": [
            "model",
            "dataset",
            "space"
          ],
          "description": "Specify lookup type; otherwise auto-detects"
        }
      },
      "required": [
        "repo_ids"
      ],
      "additionalProperties": false,
      "$schema": "http://json-schema.org/draft-07/schema#"
    }
  },
  {
    "name": "hf_doc_search",
    "description": "Search documentation about all of Hugging Face products and libraries (Transformers, Datasets, Diffusers, Gradio, Hub, and more). Use this for the most up-to-date information Returns excerpts grouped by Product and Document.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "minLength": 3,
          "maxLength": 200,
          "description": "Semantic search query"
        },
        "product": {
          "type": "string",
          "description": "Filter by Product (e.g., \"hub\", \"dataset-viewer\", \"transformers\"). Supply when known for focused results"
        }
      },
      "required": [
        "query"
      ],
      "additionalProperties": false,
      "$schema": "http://json-schema.org/draft-07/schema#"
    }
  },
  {
    "name": "hf_doc_fetch",
    "description": "Fetch a document from the Hugging Face or Gradio documentation library. For large documents, use offset to get subsequent chunks.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "doc_url": {
          "type": "string",
          "maxLength": 200,
          "description": "Documentation URL (Hugging Face or Gradio)"
        },
        "offset": {
          "type": "number",
          "minimum": 0,
          "description": "Token offset for large documents (use the offset from truncation message)"
        }
      },
      "required": [
        "doc_url"
      ],
      "additionalProperties": false,
      "$schema": "http://json-schema.org/draft-07/schema#"
    }
  },
  {
    "name": "gr1_flux1_schnell_infer",
    "description": "Generate an image using the Flux 1 Schnell Image Generator. (from evalstate/flux1_schnell)",
    "inputSchema": {
      "type": "object",
      "properties": {
        "prompt": {
          "type": "string",
          "description": "Approximately 60-70 words max - description of the image to generate."
        },
        "seed": {
          "type": "number",
          "description": "numeric value between 0 and 2147483647"
        },
        "randomize_seed": {
          "type": "boolean",
          "default": true
        },
        "width": {
          "type": "number",
          "description": "numeric value between 256 and 2048",
          "default": 1024
        },
        "height": {
          "type": "number",
          "description": "numeric value between 256 and 2048",
          "default": 1024
        },
        "num_inference_steps": {
          "type": "number",
          "description": "numeric value between 1 and 16",
          "default": 4
        }
      },
      "additionalProperties": false,
      "$schema": "http://json-schema.org/draft-07/schema#"
    }
  }
]
```

**Server**: hf-mcp-server v0.2.27
**Total Tools**: 9

#### 6.2.2 Available MCP Resources
No MCP resources specified.

#### 6.2.3 Python Server Configuration

Connection details will be configured based on server type.

#### Connection Configuration Examples

**Note: MCP servers may provide more tools than needed. Only connect to the tools listed in section 6.2.1. When setting up any connection type below, use the tool names as filters to restrict access to only the specified tools.**

Choose the appropriate connection method based on your MCP server setup:

##### Local (stdio) - For running MCP servers as subprocesses

Use this when your MCP server runs as a local subprocess. Common for development and self-hosted tools.

```python
import os
from pydantic_ai.mcp import MCPServerStdio
from dotenv import load_dotenv

load_dotenv()

# Tools allowed for this agent (from section 6.2.1)
allowed_tools = ['tool1', 'tool2', 'tool3']  # Replace with actual tool names from 6.2.1

# Example: Running a Node.js MCP server with tool filtering
stdio_server = MCPServerStdio(
    'npx',  # Command to execute
    args=['-y', '@your/mcp-package'],  # Command arguments
    timeout=10,  # Timeout for server startup
    env={'API_KEY': os.getenv('API_KEY')}  # Environment variables for the server
).filtered(
    lambda ctx, tool_def: tool_def.name in allowed_tools
)

# Example: Running a Python MCP server with tool filtering
python_server = MCPServerStdio(
    'python',
    args=['-m', 'your_mcp_server'],
    timeout=15,
    cwd='/path/to/server/directory'  # Working directory for the server
).filtered(
    lambda ctx, tool_def: tool_def.name in allowed_tools
)
```

##### Remote (HTTP) - For web-based MCP servers

Use this for MCP servers hosted remotely or accessed via HTTP endpoints. Most common for production deployments.

```python
import os
from pydantic_ai.mcp import MCPServerStreamableHTTP
from dotenv import load_dotenv

load_dotenv()

# Tools allowed for this agent (from section 6.2.1)
allowed_tools = ['tool1', 'tool2', 'tool3']  # Replace with actual tool names from 6.2.1

token = os.getenv('MCP_API_TOKEN')

http_server = MCPServerStreamableHTTP(
    url='https://your-mcp-server.com/mcp',
    headers={
        'Authorization': f'Bearer {token}',
    },
    tool_prefix='myserver',  # Prefix for tool names to avoid conflicts
    timeout=30,  # Connection timeout
    read_timeout=300,  # Read timeout for long operations
    max_retries=3  # Retry failed requests
).filtered(
    lambda ctx, tool_def: tool_def.name in allowed_tools
)
```

##### Remote (SSE) - For real-time streaming servers

Use this for MCP servers that implement the SSE (Server-Sent Events) transport method.

```python
import os
from pydantic_ai.mcp import MCPServerSSE
from dotenv import load_dotenv

load_dotenv()

# Tools allowed for this agent (from section 6.2.1)
allowed_tools = ['tool1', 'tool2', 'tool3']  # Replace with actual tool names from 6.2.1

token = os.getenv('SSE_API_TOKEN')

# Example: SSE connection to MCP server
sse_server = MCPServerSSE(
    url='https://your-server.com/mcp/sse',
    headers={
        'Authorization': f'Bearer {token}',
        'Accept': 'text/event-stream'
    },
    tool_prefix='sse_server',
    timeout=30
).filtered(
    lambda ctx, tool_def: tool_def.name in allowed_tools
)
```

##### Agent Integration Example

Once configured, use your MCP server with a Pydantic AI agent:

```python
from pydantic_ai import Agent

# Create agent with filtered MCP server as toolset
agent = Agent(
    model='claude-sonnet-4-20250514',  # or your preferred model
    toolsets=[stdio_server]  # Add your filtered MCP server (from examples above)
)

async def main():
    async with agent:  # Ensures proper cleanup of MCP connections
        result = await agent.run('Your query here')
        print(result.data)

# Run the async function
import asyncio
asyncio.run(main())
```

### 6.3 Run agent (entrypoint)
```python
async def run_agent(
    user_prompt: str,
    deps: Deps,
    message_history: list[ModelMessage] | None = None
) -> tuple[str, list[ModelMessage]]
# Code for running agent
```

## 7. Technical Design Details
### 7.1 Message History Processing
#### 7.1.1 History Processor Function
Utilize the concept of processors from Pydantic AI Framework.

```python
async def conversation_history_processor(
    messages: list[ModelMessage]
) -> list[ModelMessage]:
    # Implement message filtering/compression if needed
    # Maintain context window limits
    # Preserve essential conversation elements
    return processed_messages
```

#### 7.1.2 History Storage Strategy

Keep message history in-memory only.

### 7.2 Error Handling

#### 7.2.1 Pydantic AI Error Handling Mechanisms
##### 7.2.1.1 Built-in Exception Types
- ModelRetry: Use for recoverable errors that should trigger a retry with feedback to the model
- UserError: Runtime error for developer mistakes (not for end-user input validation)
- AgentRunError: Base class for errors during agent execution
- UsageLimitExceeded: Raised when token/cost limits are exceeded
- UnexpectedModelBehavior: Raised for unexpected model responses
- ModelHTTPError: Raised for 4xx/5xx HTTP responses from model providers
- FallbackExceptionGroup: Raised when all fallback models fail

##### 7.2.1.2 Retry Configuration
Use retry capabilities in Pydantic AI framework.
```python
# Agent-level retry configuration
Agent(
    retries=[number],  # Default retry attempts for tools
    output_retries=[number]  # Output validation retries
)
```

##### 7.2.1.3 Fallback Model Configuration
```python
from pydantic_ai.models import FallbackModel

model = FallbackModel(
    default_model='primary-model',
    'fallback-model-1',
    'fallback-model-2',
    fallback_on=(ModelHTTPError, RateLimitError)  # Exception types to trigger fallback
)
```

##### 7.2.1.4 Error Categories and Pydantic AI Handling
| Error Type | Pydantic AI Mechanism | Implementation Strategy | User Message |
|------------|----------------------|------------------------|--------------|
| Validation Error | ModelRetry with ValidationError details | Automatic retry with structured feedback | [Helpful correction guidance] |
| Tool Failure | ModelRetry exception in tool | Agent retries with context | [Explanation of issue] |
| API Timeout | ModelRetry with timeout message | Retry with exponential backoff | [Temporary issue notice] |
| Rate Limit | ModelHTTPError / UsageLimitExceeded | Fallback model or pause | [Rate limit explanation] |
| Invalid Tool Args | Automatic RetryPromptPart | Pydantic validation feedback | [Argument requirements] |
| Model Confusion | RetryPromptPart | Provide additional context | [Clarification prompt] |
| Output Mismatch | Output validators with ModelRetry | Transform or retry | [Format correction] |
| Model Unavailable | FallbackModel handling | Automatic fallback to alternate model | [Transparent to user] |

##### 7.2.1.5 Error Handling Implementation
##### 7.2.1.5.1 Output Validation
```python
@agent.output_validator
async def [validator_name](ctx: RunContext[Deps], output: OutputType):
    if [validation_condition]:
        # Request model to regenerate with guidance
        raise ModelRetry("Output doesn't meet criteria: [specific issue]")
    return output  # or transformed output
```

##### 7.2.1.5.2 Message History Error Context
- RetryPromptPart: Automatically added to message history on validation errors
    - Contains content: list[ErrorDetails] | str for error detail
    - Includes tool_name and tool_call_id when relevant

- Tool retry tracking: RunContext.retries dict tracks retry counts per tool
- Validation errors: Full Pydantic ValidationError details preserved

##### 7.2.1.5.3 Usage Limit Management
```python
# Set usage limits for the run
result = await agent.run(
    prompt,
    usage_limits=UsageLimits(
        request_limit=[max_requests],
        token_limit=[max_tokens]
    )
)
# Raises UsageLimitExceeded if exceeded
```


## 8. Testing Strategy

{{ TESTING STRATEGY CONTENT - SECTIONS 8.1 THROUGH 8.8.5 }}