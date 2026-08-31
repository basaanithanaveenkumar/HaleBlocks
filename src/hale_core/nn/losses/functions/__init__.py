"""Import token-loss implementations so `@register_token_loss` runs."""

from hale_core.nn.losses.functions import ce as _ce  # noqa: F401
from hale_core.nn.losses.functions import focal as _focal  # noqa: F401
from hale_core.nn.losses.functions import kl as _kl  # noqa: F401
from hale_core.nn.losses.functions import label_smoothing as _ls  # noqa: F401
