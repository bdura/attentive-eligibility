import requests


class Slack:

    def __init__(self, endpoint='https://hooks.slack.com/services/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'):
        """
        Wrapper that generates messages on the Slack channel defined by the *endpoint* URL.
        For more information, look at the [Slack documentation](https://api.slack.com/incoming-webhooks).

        Args:
            endpoint (str): Webhook URL.
        """
        self.endpoint = endpoint

    def send_message(self, text, attachments=None):
        """
        Sends a Slack message.

        Args:
            text (str): The message to send.
            attachments (dict): Attachements.
        Raises:
            ValueError: If the message cannot be sent.
        """

        message = {'text': text}

        if attachments is not None:
            message['attachments'] = attachments

        response = requests.post(self.endpoint, json=message)

        if response.status_code != 200:
            raise ValueError(
                'Request to slack returned an error {code}, the response is:\n{message}'.format(
                    code=response.status_code,
                    message=response.text
                )
            )
