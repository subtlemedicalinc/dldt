"""A script to build the dldt according to our need

@author: Liren Zhu <liren@subtlemedical.com>
Copyright (c) Subtle Medical, Inc. (https://subtlemedical.com)
Created on 2019/07/29
"""


import argparse
import hashlib
import logging
import os
import platform
import shutil
import sys
import sysconfig
import subprocess
import tempfile
from time import time
from typing import List, Optional
import urllib.request

from find_libpython import find_libpython

MKLML_URL = "https://github.com/intel/mkl-dnn/releases/download/v0.17/mklml_lnx_2019.0.1.20180928.tgz"
MKLML_MD5 = "a63abf155361322b9c03f8fc50f4f317"
THIS_DIR = os.path.abspath(os.path.dirname(__file__))


class BuildSubtle:
    @staticmethod
    def _check_mklml() -> bool:
        """Check if MKL-ML can be used"""
        supported_platforms = ["centos-7", "ubuntu-16.04", "ubuntu-18.04"]
        sys_sig = platform.platform().lower()
        for pf in supported_platforms:
            if pf in sys_sig:
                logging.info(
                    "MKL-ML supported for current platform %s", sys_sig
                )
                return True
        logging.warn("MKL-ML is not supported for current platofmr %s", sys_sig)
        return False

    @staticmethod
    def _download_mklml(url: str, dest: str, chunk_size: int = 1024) -> bool:
        """Download MKL-ML from url to dist_dir

        :param url: origin URL
        :param dest_dir: the destination file path
        :return: if the download was successful
        """
        logging.info("Downloading MKL-DNN package v0.17...")
        logging.info(">>> Download URL: [{}] to [{}]".format(MKLML_URL, dest))
        try:
            tic = time()
            with urllib.request.urlopen(url) as response, open(
                dest, "wb"
            ) as out_file:
                shutil.copyfileobj(response, out_file)
            toc = time()
            logging.info("Download took {:.2f} seconds".format(toc - tic))
        except Exception as exc:
            logging.warn("Downloading MKL-DNN package failed!")
            logging.warn("Failed with exc: {}".format(str(exc)))
            return False
        hasher = hashlib.new("md5")
        with open(dest, "rb") as out_file:
            chunk = out_file.read(chunk_size)
            while chunk:
                hasher.update(chunk)
                chunk = out_file.read(chunk_size)
        md5sum = hasher.hexdigest()
        logging.info(
            "Verifying md5sum... expecting [{}] got [{}]".format(
                MKLML_MD5, md5sum
            )
        )
        if MKLML_MD5 != md5sum:
            logging.warn("Checksum verification failed!")
            return False
        return True

    def _set_up_mklml(self):
        """Set up MKL-ML"""
        self._mklml_enabled = self._mklml_enabled and self._check_mklml()
        self._mklml_enabled = self._mklml_enabled and self._download_mklml(
            MKLML_URL, self._mklml_local_path
        )
        if self._mklml_enabled:
            if not self._cmake_args.endswith("\n"):
                self._cmake_args += "\n"
            self._cmake_args += "-DGEMM=MKL -DMKLROOT={}\n".format(
                self._mklml_local_path
            )

    def _set_up_python(self):
        """Set up Python"""
        if not self._python_enabled:
            return
        python_exe = sys.executable
        python_lib_path = find_libpython()
        python_include_path = ""
        try:
            python_include_path = sysconfig.get_path("include")
            if not os.path.exists(python_include_path) or not os.path.exists(
                os.path.join(python_include_path, "Python.h")
            ):
                python_include_path = ""
        except KeyError:
            pass
        if not python_exe or not python_lib_path or not python_include_path:
            logging.warn("Python setup failed!")
            logging.info("python executable: %s", python_exe)
            logging.info("python lib path: %s", python_lib_path)
            logging.info("python include path: %s", python_include_path)
            self._python_enabled = False
        else:
            if not self._cmake_args.endswith("\n"):
                self._cmake_args += "\n"
            self._cmake_args += "-DENABLE_PYTHON=ON\n"
            self._cmake_args += '-DPYTHON_EXECUTABLE="{}"\n'.format(python_exe)
            self._cmake_args += '-DPYTHON_LIBRARY="{}"\n'.format(
                python_lib_path
            )
            self._cmake_args += '-DPYTHON_INCLUDE_DIR="{}"\n'.format(
                python_include_path
            )

    def __init__(self, args: Optional[List[str]] = None):
        parser = argparse.ArgumentParser(
            description="the build script for OpenVINO/dldt"
        )
        parser.add_argument(
            "--cmake-command",
            "-c",
            default="cmake",
            help="the name or path to the cmake command",
        )
        parser.add_argument(
            "--make-command",
            "-m",
            default="make",
            help="the name or path to the make command",
        )
        run_args = parser.parse_args(args)
        self._cmake_command = run_args.cmake_command
        self._make_command = run_args.make_command
        self._build_dir = tempfile.TemporaryDirectory()
        self._download_dir = tempfile.TemporaryDirectory()
        self._mklml_enabled = True
        self._mklml_local_path = os.path.join(
            self._download_dir.name, "mklml_lnx.tgz"
        )
        self._python_enabled = True
        self._cmake_args = """
-DCMAKE_BUILD_TYPE=Release
-DBUILD_TESTING=OFF
-DENABLE_CLDNN=OFF
-DENABLE_GNA=OFF
-DENABLE_MYRIAD=OFF
-DENABLE_OPENCV=OFF
-DENABLE_VPU=OFF
"""
        self._set_up_mklml()
        self._set_up_python()

    def build_inference_engine(self):
        command = " ".join(
            [self._cmake_command]
            + self._cmake_args.split()
            + [os.path.join(THIS_DIR, "../inference-engine")]
        )
        logging.info("Command to run: %s", command)
        proc = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=self._build_dir.name,
        )
        if proc.returncode != 0:
            logging.error("cmake command failed!")
        logging.info(proc.stdout.decode())
        command = self._make_command + " -j2"
        logging.info("Command to run: %s", command)
        proc = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=self._build_dir.name,
        )
        if proc.returncode != 0:
            logging.error("make command failed!")
        logging.info(proc.stdout.decode())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    builder = BuildSubtle()
    builder.build_inference_engine()
