import llm
import click
from typing import Dict, Any, Type, List
import yaml
from pydantic import BaseModel, create_model
import sys


def resolve_ref(ref: str, schema_dict: Dict) -> Dict:
    """Resolve a $ref reference in the schema."""
    if not ref.startswith("#/"):
        raise ValueError(f"Only local references are supported: {ref}")

    path = ref[2:].split("/")
    current = schema_dict
    for part in path:
        if part not in current:
            raise ValueError(f"Invalid reference: {ref}")
        current = current[part]
    return current


def create_models_from_schema(schema_dict: Dict) -> Dict[str, Type[BaseModel]]:
    """Create all models defined in the schema."""
    models = {}

    def parse_schema_to_pydantic_model(
        schema_dict: Dict, model_name: str = None
    ) -> Type[BaseModel]:
        """Parse a schema into a Pydantic model, supporting nested objects and references."""
        # If it's a reference, resolve it
        if isinstance(schema_dict, dict) and "$ref" in schema_dict:
            ref_path = schema_dict["$ref"][2:].split("/")
            if ref_path[0] in models:
                return models[ref_path[0]]
            referenced_schema = {
                ref_path[0]: resolve_ref(schema_dict["$ref"], root_schema)
            }
            return parse_schema_to_pydantic_model(referenced_schema)

        if len(schema_dict) == 1 and isinstance(next(iter(schema_dict.values())), dict):
            model_name = next(iter(schema_dict.keys()))
            fields_dict = next(iter(schema_dict.values()))

            # Handle array type with items
            if (
                isinstance(fields_dict, dict)
                and fields_dict.get("type") == "array"
                and "items" in fields_dict
            ):
                item_model = parse_schema_to_pydantic_model(fields_dict["items"])
                # Create a wrapper model for arrays to satisfy OpenAI's object requirement
                model = create_model(model_name, items=(List[item_model], ...))
                models[model_name] = model
                return model

            # Map string type names to actual types
            type_mapping = {
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": List,
                "dict": Dict,
            }

            # Convert fields
            fields: Dict[str, Any] = {}
            for field_name, field_type in fields_dict.items():
                if isinstance(field_type, dict):
                    # Nested object
                    nested_model = parse_schema_to_pydantic_model(
                        field_type, field_name
                    )
                    fields[field_name] = (nested_model, ...)
                elif isinstance(field_type, str):
                    python_type = type_mapping.get(field_type, Any)
                    fields[field_name] = (python_type, ...)

            model = create_model(model_name, **fields)
            models[model_name] = model
            return model

        return create_model(model_name or "DynamicModel", **{})

    root_schema = schema_dict
    # First pass: create all top-level models
    for model_name, model_schema in schema_dict.items():
        if model_name not in models:
            parse_schema_to_pydantic_model({model_name: model_schema})

    return models


@llm.hookimpl
def register_commands(cli):
    """Register the structure command with the CLI."""

    @cli.command()
    @click.argument("prompt", type=click.STRING)
    @click.option("--schema", help="JSON schema to parse against", required=True)
    @click.option(
        "-m", "--model", default="gpt-4o", help="Model to use for structured output"
    )
    def structure(prompt: str, schema: str, model: str):
        """Generate structured output based on a JSON schema.

        The output will be validated against the provided JSON schema.
        The model must support JSON mode/structured output.
        """

        if prompt == "-":
            # Read from stdin
            prompt = sys.stdin.read()

        try:
            # Load schema from file
            with open(schema) as f:
                schema_dict = yaml.safe_load(f)

            # Create dynamic Pydantic models from schema
            models = create_models_from_schema(schema_dict)

            # For array types, we want the container model
            model_name = next(iter(schema_dict.keys()))
            DynamicModel = models[model_name]

            # Get the model instance
            model_instance = llm.get_model(model)
            model_id = model_instance.model_id

            # Supported models per: https://platform.openai.com/docs/guides/structured-outputs/examples#supported-schemas
            # o1-2024-12-17 and later
            # gpt-4o-mini-2024-07-18 and later
            # gpt-4o-2024-08-06 and later

            # TODO: Make supported models detection more robust
            # TODO: Add Gemini native support
            # TODO: Add instructor for all other models

            if ("o1" in model_id or "gpt-4o" in model_id) and model_id != "gpt-4":
                client = model_instance.get_client()
                schema = DynamicModel.model_json_schema()
                response = client.beta.chat.completions.parse(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=DynamicModel,
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