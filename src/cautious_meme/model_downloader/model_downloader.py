import venv
import sys
import subprocess
import importlib.resources
import tomllib
from uuid import uuid4
from pathlib import Path
from typing import Any


class VenvManager:
    def __init__(self, prefix: str) -> None:
        self.prefix: str = prefix

        pre_existing_venv: str | None = self._find_existing_venv()

        self.venv_dir: str = pre_existing_venv if pre_existing_venv is not None \
            else self._generate_venv_dir_name()

    def get_venv_dir(self: "VenvManager") -> str:
        return self.venv_dir

    def get_prefix(self: "VenvManager") -> str:
        return self.prefix

    def create_venv(self: "VenvManager") -> None:
        # create virtual environment
        venv_builder: venv.EnvBuilder = venv.EnvBuilder(with_pip=True)
        venv_builder.create(self.venv_dir)

    def remove_venv(self: "VenvManager") -> None:
        VenvManager._rmdir(Path(self.venv_dir))

    def _find_existing_venv(self: "VenvManager") -> str | None:
        for possible_env in Path.cwd().glob(f".{self.prefix}_??????????????"):
            if possible_env.is_dir():
                return possible_env.as_posix()
        return None

    def _generate_venv_dir_name(self: "VenvManager") -> str:
        return f".{self.prefix}_{
        uuid4().urn
        .split(":")
        .pop()
        .replace("-", "")[:14]
        }"

    @classmethod
    def _rmdir(cls: "VenvManager", dir_path: Path) -> None:
        for item in dir_path.glob("*"):
            if item.is_dir():
                VenvManager._rmdir(item)
            else:
                item.unlink()
        dir_path.rmdir()


# Download dependencies in an existing virtual environment and add the dependencies to the module path
def bootstrap_model_downloader(venv_dir: str, dependencies: list[str]) -> None:
    # download dependencies
    subprocess.check_call([f"{venv_dir}/bin/pip", "install"] + dependencies, shell=False)
    venv_modules: Path | None = next(Path(f"{venv_dir}/lib/").glob("python3.*/site-packages"), None)

    if venv_modules is not None:
        sys.path.insert(0, venv_modules.as_posix())
    else:
        raise RuntimeError


def load_config_from_resources() -> dict[str,Any]:
    config_text: str = importlib.resources.read_text("cautious_meme.model_downloader", "config.toml")
    return tomllib.loads(config_text)


def orchestrate_model_download(argv: list[str] = None) -> int:
    config: dict[str,Any] = load_config_from_resources()

    venv_manager: VenvManager = VenvManager("model_downloader")
    venv_manager.create_venv()

    exit_code: int = 0
    for model_repo_config in config["model-repo"]:
        try:
                bootstrap_model_downloader(venv_manager.get_venv_dir(), model_repo_config["repo-provider"]["dependencies"])
        except subprocess.CalledProcessError | BaseException as e:
            print(
                f"Unable to download dependencies for {model_repo_config["repo-provider"]["provider"]} or " +
                "unable to access the dependency modules in virtual " +
                f"environment: {venv_manager.get_venv_dir()}"
            )
            return 1

        # TODO: Will need to import modules based on repo-provider which will mean moving out concrete repo_interface implementations to separate files
        # import repo_interface
        import repo_interface
        # call and return repo_interface.main(args)
        exit_code: int = exit_code + repo_interface.download_model(model_repo_config)
        # delete temporary environment used for pulling AI model
        venv_manager.remove_venv()

    return exit_code

if __name__ == "__main__":
    sys.exit(orchestrate_model_download(sys.argv))