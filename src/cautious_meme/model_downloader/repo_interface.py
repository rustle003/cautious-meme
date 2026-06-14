import sys
import tomllib
import huggingface_hub
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Any, Type, override, Callable, Union


@dataclass(frozen=True)
class RepoConfig:
    repo_id: str
    repo_provider: str
    repo_file: str
    download_path: str
    revision: str

    @classmethod
    def from_toml(cls: Type["RepoConfig"], config: dict[str,Any]) -> "RepoConfig":
        return cls(
            repo_id=config["repo_id"],
            repo_provider=config["repo-provider"]["provider"],
            repo_file=config["repo_file"] if "repo_file" in config else "",
            download_path=config["download_path"] if "download_path" in config else "",
            revision=config["revision"] if "revision" in config else ""
        )


class RepoInterface(Protocol):
    def clone_repo(self: "RepoInterface", repo_id: str, download_path: str = "", revision ="") -> None: ...
    def pull_file(self: "RepoInterface", repo_id: str, file_name: str, download_path: str = "", revision ="") -> None: ...
    def validate_configuration(cls: Type["RepoInterface"], config: RepoConfig) -> None: ...


class RepoFetchResolver[T : RepoInterface]:
    def __init__(self: "RepoFetchResolver", repo: T, repo_config: RepoConfig) -> None:
        self.pull_repo_data: Union[Callable[[],None], Callable[[],None]]= \
            (lambda : repo.clone_repo(
                repo_config.repo_id,
                repo_config.download_path,
                repo_config.revision
            )) \
            if repo_config.repo_file == "" \
            else (lambda : repo.pull_file(
                repo_config.repo_id,
                repo_config.repo_file,
                repo_config.download_path,
                repo_config.revision
            ))


class HuggingFaceRepoInterface(RepoInterface):
    def __init__(self: "HuggingFaceRepoInterface"):
        pass

    @override
    def clone_repo(self: RepoInterface, repo_id: str, download_path: str = "", revision: str = "") -> None:
        huggingface_hub.snapshot_download(
            repo_id=repo_id,
            local_dir=download_path if download_path != "" else None,
            revision=revision if revision != "" else None,
        )

    @override
    def pull_file(self: "RepoInterface", repo_id: str, file_name: str, download_path: str = "", revision ="") -> None:
        huggingface_hub.hf_hub_download(
            repo_id=repo_id,
            local_dir=download_path if download_path != "" else None,
            revision=revision if revision != "" else None,
            filename=file_name
        )

    @override
    def validate_configuration(cls: Type["RepoInterface"], config: RepoConfig) -> None:
        pass


class RepoFactory:
    class UnknownRepoProviderException(Exception):
        pass

    def __init__(self: "RepoFactory", repo_config: RepoConfig) -> None:
        self.repo_config: RepoConfig = repo_config

    def build_repo_interface[T : RepoInterface](self: "RepoFactory") -> RepoFetchResolver[T]:
        """Create concrete repo interface if repo provider is known

        :raises UnknownRepoProviderException: if the repo provider is unknown
        :return RepoFetchResolver[T]: if the repo provider is valid:
        """
        match self.repo_config.repo_provider:
            case "huggingface":
                repo_interface: RepoInterface = HuggingFaceRepoInterface()
                repo_interface.validate_configuration(self.repo_config)
                return RepoFetchResolver(repo_interface, self.repo_config)
            case _:
                raise RepoFactory.UnknownRepoProviderException(
                    f"Unknown repo provider: {self.repo_config.repo_provider}. Known repo providers are: 'huggingface'."
                )


def download_model(config: dict[str,Any]) -> int:
    repo_config: RepoConfig = RepoConfig.from_toml(config)
    repo_factory: RepoFactory = RepoFactory(repo_config)
    repo_interface: RepoFetchResolver[RepoInterface] = repo_factory.build_repo_interface()
    repo_interface.pull_repo_data()

    return 0