class ServiceError(Exception):
    """A public, credential-free error message and its HTTP status."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code
