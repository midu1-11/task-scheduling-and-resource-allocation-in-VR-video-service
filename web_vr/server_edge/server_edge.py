#!/usr/bin/env python3

import socket
import os
import stat
from urllib.parse import unquote
import time
import cv2
import base64
from image_process import *
import multiprocessing

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

class EdgeServer:
    def __init__(self, host="localhost", port=9001, directory="."):
        print(f"Server started. Listening at http://{host}:{port}/")

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

    def accept(self):
        while True:
            (client, address) = self.sock.accept()
            process = multiprocessing.Process(target=HTTPServer().accept_request,args=(client, address))
            process.start()

    def teardown_socket(self):
        if self.sock is not None:
            self.sock.shutdown()
            self.sock.close()

# 实现GET和POST requests的HTTP server。
class HTTPServer:
    """
    Our actual HTTP server which will service GET and POST requests.
    """

    def __init__(self, host="localhost", port=9001, directory="."):
        pass
        # print(f"Server started. Listening at http://{host}:{port}/")
        # self.videoLoader = VideoLoader()
        # self.host = host
        # self.port = port
        # self.working_dir = directory

        # self.setup_socket()
        # self.accept()
        #
        # self.teardown_socket()


    # def setup_socket(self):
    #     self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #     self.sock.bind((self.host, self.port))
    #     self.sock.listen(128)
    #
    # def teardown_socket(self):
    #     if self.sock is not None:
    #         self.sock.shutdown()
    #         self.sock.close()

    # def accept(self):
    #     (client, address) = self.sock.accept()
    #     self.accept_request()
    #     # process = multiprocessing.Process(target=self.accept_request, args=(client, address))
    #     # process.start()
    #     # th = Thread(target=self.accept_request, args=(client, address))
    #     # th.start()
    #     # self.accept_request(client, address)


    def accept_request(self, client_sock, client_addr):
        print(os.getpid())

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

        # if (not os.path.exists(requested_file)):
        #     return self.resource_not_found()
        # elif (not has_permission_other(requested_file)):
        #     return self.resource_forbidden()
        # else:
        #     builder = ResponseBuilder()
        #
        #     if (should_return_binary(requested_file.split(".")[1])):
        #         # builder.set_content(get_file_binary_contents(requested_file))
        #         builder.my_set_content(self.videoLoader,requested_file)
        #     else:
        #         builder.set_content(get_file_contents(requested_file))
        #
        #     builder.set_status("200", "OK")
        #
        #     builder.add_header("Access-Control-Allow-Headers","Content-Type")
        #     builder.add_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        #     builder.add_header("Access-Control-Allow-Origin", "*")
        #     # builder.add_header("Cache-Control", "no-cache")
        #     builder.add_header("Content-Type", "text/plain; charset=utf-8")
        #     # builder.add_header("Expires", "-1")
        #     # builder.add_header("Pragma", "no-cache")
        #     # builder.add_header("Server", "Microsoft-IIS/10.0")
        #     # builder.add_header("X-AspNet-Version", "4.0.30319")
        #     # builder.add_header("X-Powered-By", "ASP.NET")
        #
        #     return builder.build()

        builder = ResponseBuilder()

        if (should_return_binary(requested_file.split(".")[1])):
            # builder.set_content(get_file_binary_contents(requested_file))
            builder.my_set_content(VideoLoader(), requested_file)
        else:
            builder.set_content(get_file_contents(requested_file))

        builder.set_status("200", "OK")

        builder.add_header("Access-Control-Allow-Headers", "Content-Type")
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

    def my_set_content(self,videoLoader,requested_file):
        self.content = b"data:image/jpeg;base64,"+videoLoader.fetchAframe(requested_file)

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
        # self.cap_pose = cv2.VideoCapture('pose.mp4')
        # self.cap_hand = cv2.VideoCapture('hand.mp4')
        # self.cap_face = cv2.VideoCapture('face.mp4')
        # self.cap_video = cv2.VideoCapture('video.mp4')

        self.image_processor = image_processor()

    def fetchAframe(self,requested_file):
        type = None

        if requested_file == "pose.html":
            type = "pose"
            img = cv2.VideoCapture('pose.mp4').read()[1]
        elif requested_file == "face.html":
            type = "face"
            img = cv2.VideoCapture('face.mp4').read()[1]
        elif requested_file == "hand.html":
            type = "hand"
            img = cv2.VideoCapture('hand.mp4').read()[1]
        elif requested_file == "artwork.html":
            type = "artwork"
            img = cv2.VideoCapture('video.mp4').read()[1]
        elif requested_file == "blur.html":
            type = "blur"
            img = cv2.VideoCapture('video.mp4').read()[1]
        elif requested_file == "sharp.html":
            type = "sharp"
            img = cv2.VideoCapture('video.mp4').read()[1]

        img = self.image_processor.process_one_img(img,type)

        base64_str = cv2.imencode('.jpg', img)[1].tostring()
        base64_byte = base64.b64encode(base64_str)

        return base64_byte



def uuu():
    print("kkkkkkkkk")
if __name__ == "__main__":
    EdgeServer()
    # HTTPServer()
    # for i in range(3):
    #     process = multiprocessing.Process(target=uuu)
    #     process.start()