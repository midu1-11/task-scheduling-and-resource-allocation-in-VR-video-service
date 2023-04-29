#!/usr/bin/env python3

import socket
import os
import stat
from urllib.parse import unquote
import time
import cv2
import base64

from threading import Thread

# Equivalent to CRLF, named NEWLINE for clarity
NEWLINE = "\r\n"


# Let's define some functions to help us deal with files, since reading them
# and returning their data is going to be a very common operation.

def get_file_contents(file_name):
    """Returns the text content of `file_name`"""
    with open(file_name, "r") as f:
        return f.read()


def get_file_binary_contents(file_name):
    """Returns the binary content of `file_name`"""
    with open(file_name, "rb") as f:
        return f.read()


def has_permission_other(file_name):
    """Returns `True` if the `file_name` has read permission on other group
    In Unix based architectures, permissions are divided into three groups:
    1. Owner
    2. Group
    3. Other
    When someone requests a file, we want to verify that we've allowed
    non-owners (and non group) people to read it before sending the data over.
    """
    stmode = os.stat(file_name).st_mode
    return getattr(stat, "S_IROTH") & stmode > 0


# Some files should be read in plain text, whereas others should be read
# as binary. To maintain a mapping from file types to their expected form, we
# have a `set` that maintains membership of file extensions expected in binary.
# We've defined a starting point for this set, which you may add to as necessary.
# TODO: Finish this set with all relevant files types that should be read in binary
binary_type_files = set(["jpg", "jpeg", "mp3", "png", "html", "js", "css"])


def should_return_binary(file_extension):
    """
    Returns `True` if the file with `file_extension` should be sent back as
    binary.
    """
    return file_extension in binary_type_files


# For a client to know what sort of file you're returning, it must have what's
# called a MIME type. We will maintain a `dictionary` mapping file extensions
# to their MIME type so that we may easily access the correct type when
# responding to requests.
# TODO: Finish this dictionary with all required MIME types
mime_types = {
    "html": "text/html",
    "css": "text/css",
    "js": "text/javascript",
    "mp3": "audio/mpeg",
    "png": "image/png",
    "jpg": "image/jpg",
    "jpeg": "image/jpeg"
}


def get_file_mime_type(file_extension):
    """
    Returns the MIME type for `file_extension` if present, otherwise
    returns the MIME type for plain text.
    """
    mime_type = mime_types[file_extension]
    return mime_type if mime_type is not None else "text/plain"


# 实现GET和POST requests的HTTP server。
class HTTPServer:
    """
    Our actual HTTP server which will service GET and POST requests.
    """

    def __init__(self, host="10.0.24.7", port=9001, directory="."):
        print(f"Server started. Listening at http://{host}:{port}/")
        self.videoLoader = VideoLoader()
        self.host = host
        self.port = port
        self.working_dir = directory

        self.setup_socket()
        self.accept()

        self.teardown_socket()


    def setup_socket(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.host, self.port))
        self.sock.listen(128)

    def teardown_socket(self):
        if self.sock is not None:
            self.sock.shutdown()
            self.sock.close()

    def accept(self):
        while True:
            (client, address) = self.sock.accept()
            th = Thread(target=self.accept_request, args=(client, address))
            th.start()
            # self.accept_request(client, address)


    def accept_request(self, client_sock, client_addr):
        data = client_sock.recv(4096)
        req = data.decode("utf-8")

        start_time = time.time()

        # print(os.system(r".\abc.bat"))

        response = self.process_response(req)
        client_sock.send(response)

        # clean up
        client_sock.shutdown(1)
        client_sock.close()

        end_time = time.time()
        interval = end_time - start_time
        print("time:"+str(interval))

    def process_response(self, request):
        formatted_data = request.strip().split(NEWLINE)
        request_words = formatted_data[0].split()

        if len(request_words) == 0:
            return

        requested_file = request_words[1][1:]
        if request_words[0] == "GET":
            return self.get_request(requested_file, formatted_data)
        if request_words[0] == "POST":
            return self.post_request(requested_file, formatted_data)
        return self.method_not_allowed()

    # The response to a HEADER request
    def head_request(self, requested_file, data):
        if not os.path.exists(requested_file):
            response = NOT_FOUND
        elif not has_permission_other(requested_file):
            response = FORBIDDEN
        else:
            response = OK

        return response.encode('utf-8')

    # TODO: Write the response to a GET request

    def get_request(self, requested_file, data):

        if (not os.path.exists(requested_file)):
            return self.resource_not_found()
        elif (not has_permission_other(requested_file)):
            print("forbid")
            return self.resource_forbidden()
        else:
            builder = ResponseBuilder()

            if (should_return_binary(requested_file.split(".")[1])):
                # builder.set_content(get_file_binary_contents(requested_file))
                builder.set_content1(self.videoLoader)
            else:
                builder.set_content(get_file_contents(requested_file))

            builder.set_status("200", "OK")

            builder.add_header("Access-Control-Allow-Headers","Content-Type")
            builder.add_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
            builder.add_header("Access-Control-Allow-Origin", "*")
            # builder.add_header("Cache-Control", "no-cache")
            builder.add_header("Content-Type", "text/plain; charset=utf-8")
            # builder.add_header("Expires", "-1")
            # builder.add_header("Pragma", "no-cache")
            # builder.add_header("Server", "Microsoft-IIS/10.0")
            # builder.add_header("X-AspNet-Version", "4.0.30319")
            # builder.add_header("X-Powered-By", "ASP.NET")

            return builder.build()

        """
        Responds to a GET request with the associated bytes.
        If the request is to a file that does not exist, returns
        a `NOT FOUND` error.
        If the request is to a file that does not have the `other`
        read permission, returns a `FORBIDDEN` error.
        Otherwise, we must read the requested file's content, either
        in binary or text depending on `should_return_binary` and
        send it back with a status set and appropriate mime type
        depending on `get_file_mime_type`.
        """

    # TODO: Write the response to a POST request
    def post_request(self, requested_file, data):

        builder = ResponseBuilder()
        builder.set_status("200", "OK")
        builder.add_header("Connection", "close")
        builder.add_header("Content-Type", mime_types["html"])
        builder.set_content(get_file_contents("MyForm.html"))
        return builder.build()

    def method_not_allowed(self):
        """
        Returns 405 not allowed status and gives allowed methods.
        TODO: If you are not going to complete the `ResponseBuilder`,
        This must be rewritten.
        """
        builder = ResponseBuilder()
        builder.set_status("405", "METHOD NOT ALLOWED")
        allowed = ", ".join(["GET", "POST"])
        builder.add_header("Allow", allowed)
        builder.add_header("Connection", "close")
        return builder.build()

    # TODO: Make a function that handles not found error
    def resource_not_found(self):
        """
        Returns 404 not found status and sends back our 404.html page.
        """
        builder = ResponseBuilder()
        builder.set_status("404", "NOT FOUND")
        builder.add_header("Connection", "close")
        builder.add_header("Content-Type", mime_types["html"])
        builder.set_content(get_file_contents("404.html"))
        return builder.build()

    # TODO: Make a function that handles forbidden error
    def resource_forbidden(self):
        """
        Returns 403 FORBIDDEN status and sends back our 403.html page.
        """
        builder = ResponseBuilder()
        builder.set_status("403", "FORBIDDEN")
        builder.add_header("Connection", "close")
        builder.add_header("Content-Type", mime_types["html"])
        builder.set_content(get_file_contents("403.html"))
        return builder.build()


# 写了一个ResponseBuilder来创建出正确格式的response message。
class ResponseBuilder:
    """
    This class is here for your use if you want to use it. This follows
    the builder design pattern to assist you in forming a response. An
    example of its use is in the `method_not_allowed` function.
    Its use is optional, but it is likely to help, and completing and using
    this function to build your responses will give 5 bonus points.
    """

    def __init__(self):
        """
        Initialize the parts of a response to nothing.
        """
        self.headers = []
        self.status = None
        self.content = None

    def add_header(self, headerKey, headerValue):
        """ Adds a new header to the response """
        self.headers.append(f"{headerKey}: {headerValue}")

    def set_status(self, statusCode, statusMessage):
        """ Sets the status of the response """
        self.status = f"HTTP/1.1 {statusCode} {statusMessage}"

    def set_content(self, content):
        """ Sets `self.content` to the bytes of the content """
        if isinstance(content, (bytes, bytearray)):
            self.content = content
        else:
            self.content = content.encode("utf-8")

    def set_content1(self,videoLoader):
        # self.content = b'data:image/jpeg;base64,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'
        self.content = b"data:image/jpeg;base64,"+videoLoader.fetchAframe()

    # TODO Complete the build function
    def build(self):

        response = self.status
        response += NEWLINE
        for i in self.headers:
            response += i
            response += NEWLINE
        # response += NEWLINE
        response += NEWLINE
        response = response.encode("utf-8")
        response += self.content

        return response
        """
        Returns the utf-8 bytes of the response.
        Uses the `self.status`, `self.headers` and `self.content` to form
        an HTTP response in valid formatting per w3c specifications, which
        can be seen here:
          https://www.w3.org/Protocols/rfc2616/rfc2616-sec6.html
        or here:
          https://www.tutorialspoint.com/http/http_responses.htm
        Where CRLF is our `NEWLINE` constant.
        """


class VideoLoader:
    def __init__(self):
        self.cap = cv2.VideoCapture('dance.mp4')
        # self.frameCount = 0

    def fetchAframe(self):
        # self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frameCount)
        # print("frame:" + str(self.cap.get(1)))
        # print(self.frameCount)

        success, img = self.cap.read()
        # self.frameCount += 1

        base64_str = cv2.imencode('.jpg', img)[1].tostring()
        base64_byte = base64.b64encode(base64_str)
        return base64_byte




if __name__ == "__main__":
    HTTPServer()