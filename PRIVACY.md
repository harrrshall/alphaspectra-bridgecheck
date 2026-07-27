# Privacy

The public static browser application performs inference locally in the browser. It does not upload
spectra, set analytics cookies or contact an inference API. The browser fetches only the application
assets and public model artifact from the same GitHub Pages origin.

The optional FastAPI server is self-hosted. It processes request bodies in memory and the reference
implementation does not log spectra or persist requests. Operators are responsible for their own
network, access, retention and privacy policy.
