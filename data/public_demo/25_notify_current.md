# Pulse Notifications — Current Delivery Policy

Status: current. Pulse retries transient delivery failures five times using exponential backoff capped at 15 minutes. Notification code NT-5502 means the destination rejected authentication. Customer-visible alerts use the verified sender alerts@northstar.example.
