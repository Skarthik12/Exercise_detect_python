from azure.storage.blob import BlobServiceClient
from io import BytesIO

class BlobUploader:
    def __init__(self, connection_string, container_name):
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service_client.get_container_client(container_name)

    def upload_video(self, video_stream, blob_name):
        """
        Upload a video stream directly to Azure Blob Storage.

        :param video_stream: The video stream to be uploaded (e.g., BytesIO object).
        :param blob_name: The name of the blob to be created in the container.
        """
        # Ensure the stream is at the beginning
        if isinstance(video_stream, BytesIO):
            video_stream.seek(0)

        try:
            # Upload the video stream
            self.container_client.upload_blob(name=blob_name, data=video_stream, blob_type="BlockBlob", overwrite=True)
            print(f"Video uploaded successfully as {blob_name}")
        except Exception as e:
            print(f"Failed to upload video: {e}")

# Usage example
if __name__ == "__main__":
    connection_string = ""
    container_name = "dev"
    blob_name = "Camfusion"

    # Example video stream (using BytesIO for demonstration purposes)
    video_stream = BytesIO(b"sample video data")

    uploader = BlobUploader(connection_string, container_name)
    uploader.upload_video(video_stream, blob_name)
