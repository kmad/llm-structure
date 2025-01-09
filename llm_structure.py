import llm
import click
from typing import Dict, Any, Type, List
import yaml
from pydantic import BaseModel, create_model

def parse_schema_to_pydantic_model(schema_dict: Dict) -> Type[BaseModel]:
    # If the schema is in the format {'ModelName': {'field': 'type', ...}}
    if len(schema_dict) == 1 and isinstance(next(iter(schema_dict.values())), dict):
        model_name = next(iter(schema_dict.keys()))
        fields_dict = next(iter(schema_dict.values()))
        
        # Map string type names to actual types
        type_mapping = {
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': List,
            'dict': Dict
        }
        
        # Convert fields
        fields: Dict[str, Any] = {}
        for field_name, field_type in fields_dict.items():
            python_type = type_mapping.get(field_type, Any)
            fields[field_name] = (python_type, ...)
        
        return create_model(model_name, **fields)
    
    # Original JSON Schema format handling
    properties = schema_dict.get("properties", {})
    required = schema_dict.get("required", [])
    
    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": List,
        "object": Dict
    }
    
    fields: Dict[str, Any] = {}
    for field_name, field_schema in properties.items():
        field_type = type_mapping.get(field_schema.get("type"), Any)
        default = ... if field_name in required else None
        fields[field_name] = (field_type, default)
    
    return create_model(schema_dict.get("title", "DynamicModel"), **fields)

@llm.hookimpl
def register_commands(cli):
    """Register the structure command with the CLI."""

    @cli.command()
    @click.argument("prompt")
    @click.option("--schema", help="JSON schema to parse against", required=True)
    @click.option(
        "-m", "--model", default="gpt-4", help="Model to use for structured output"
    )
    def structure(prompt: str, schema: str, model: str):
        """Generate structured output based on a JSON schema.

        The output will be validated against the provided JSON schema.
        The model must support JSON mode/structured output.
        """
        try:
            # Load schema from file
            with open(schema) as f:
                schema_dict = yaml.safe_load(f)
            
            # Create dynamic Pydantic model from schema
            DynamicModel = parse_schema_to_pydantic_model(schema_dict)
            
            # Get the model instance
            model_instance = llm.get_model(model)
            model_id = model_instance.model_id
            
            if ("o1" in model_id or "gpt-4o" in model_id) and model_id != "gpt-4":
                client = model_instance.get_client()
                schema = DynamicModel.model_json_schema()
                response = client.beta.chat.completions.parse(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=DynamicModel
                )
                result = response.choices[0].message.content
                if result:
                    print(result)   
                else:
                    raise click.BadParameter("No response from model")
            else:
                raise click.BadParameter("Model does not support structured output")

        except Exception as e:
            click.echo(f"Error: {str(e)}", err=True)
            raise click.Abort()
