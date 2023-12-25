import warnings

from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog


class ProjectHooks:
    @hook_impl
    def after_catalog_created(self, catalog: DataCatalog) -> None:
        # Skorecard has a warning about a future change in pandas about inf values
        # which spams the logs and is not relevant.
        warnings.filterwarnings("ignore", category=FutureWarning, module="skorecard")

        # Optbinning complains about invalue values encountered in the casting of data
        # which also appears to not be relevant.
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="optbinning")
