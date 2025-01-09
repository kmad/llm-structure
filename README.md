# LLM Structured Outputs

<!-- [![PyPI version](https://badge.fury.io/py/llm-structure.svg)](https://badge.fury.io/py/llm-structure)
[![Python](https://img.shields.io/pypi/pyversions/llm-structure.svg)](https://pypi.org/project/llm-structure/) -->
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A plugin for the [llm](https://github.com/simonw/llm) CLI tool that enables structured output generation from language models according to a specified schema. For usability, users can define their desired schema in YAML.

## Features

- 🔧 Generate structured outputs according to a simple YAML definition
- 🔄 Support for nested objects and references
- 🤖 Uses OpenAI's native structured output API mode
- 🛡️ Built-in validation using Pydantic

## Installation

Assuming you have `llm` installed already, simply run:

```bash
llm install -U llm-structure
```

## Usage

### Schema Definition

Create a YAML file defining your schema. Note the use of `$ref` which allows for lists of objects in this case. This is the same as defining `List[People]`.


```yaml
# people.yaml

People:
  type: array
  items:
    $ref: "#/Person"
Person:
  name: str
  age: int
```

Then run the plugin:
```bash
llm structure "Chase, 53, and 49 year old Mia went to the market." --schema people.yaml -m gpt-4o
```
Alternatively, you can use `STDIN` by specifying `-` as the prompt:

```bash
echo "Chase, 53, and 49 year old Mia went to the market." | llm structure - --schema people.yaml -m gpt-4o
```

This prints output directly to the console in JSON format:

```json
{
    "items": [
        {
            "name": "Chase",
            "age": 53
        },
        {
            "name": "Mia",
            "age": 49
        }
    ]
}
```



## Supported Models

Currently supports OpenAI models:
  - o1-2024-12-17 and later
  - gpt-4o-mini-2024-07-18 and later
  - gpt-4o-2024-08-06 and later

Integration with Gemini (native) and instructor for the balance of models is coming soon.

## Requirements

- Python 3.8+
- llm 
- PyYAML
- Pydantic
