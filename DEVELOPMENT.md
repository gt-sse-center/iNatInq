# DEVELOPMENT

## Installing Dependencies

We use [uv workspaces](https://docs.astral.sh/uv/concepts/projects/workspaces/) to manage all the python dependencies.

There is a top-level `pyproject.toml`, and each directory corresponds to a sub-project which has its own `pyproject.toml`.

This makes dependency management across the various sub-projects quite easy. To sync all the dependencies across the workspace, we can run

```sh
uv sync --all-packages
```

This is by far the most convenient as it allows full compatibility with IDE tools such as VS Code.

### Install Sub-project Dependencies

To install the dependencies for a specific sub-project, you can run

```sh
uv sync --package <sub-project>
```

This means, that if you want to install the dependencies for the `iNatInqPer` project, you can run

```sh
uv sync --package inatinqperf  # It is case insensitive.
```

## Running Tests
