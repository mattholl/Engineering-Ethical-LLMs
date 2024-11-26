import os
import tarfile
import boto3
from botocore.exceptions import NoCredentialsError
from typing import Optional


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


def untar_file(tar_filename: str):
    # If the extracted tarfile exists then don't bother doing the work.
    archive = tarfile.open(tar_filename, "r:gz")

    # Main directory
    main_archive_dir = os.path.commonprefix(archive.getnames())
    print(f"Archive folder name: {main_archive_dir}")

    if os.path.exists(os.path.join(os.getcwd(), main_archive_dir)):
        print(f"File exists: {main_archive_dir}")
        return main_archive_dir

    # For debugging the full archive contents can be printed
    # for tarinfo in tar:
    #     print(tarinfo.name, "is", tarinfo.size, "bytes in size and is ", end="")
    #     if tarinfo.isreg():
    #         print("a regular file.")
    #     elif tarinfo.isdir():
    #         print("a directory.")
    #     else:
    #         print("something else.")

    print(f"Extracting: {tar_filename}")
    archive.extractall()
    archive.close()

    return main_archive_dir


def upload_to_aws(local_file, bucket, s3_file):
    s3 = boto3.client("s3")

    try:
        s3.upload_file(local_file, bucket, s3_file)
        print("Upload Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False


def get_file_from_aws(bucket_name: str, folder_name: str, local_directory: str = "."):
    if local_directory:
        if os.path.exists(
            os.path.join(os.getcwd(), f"{local_directory}/{folder_name}")
        ):
            print(f"File exists: {local_directory}/{folder_name}")
            return

    if os.path.exists(os.path.join(os.getcwd(), folder_name)):
        print(f"File exists: {folder_name}")
        return

    # If the file already exists then no need to overwrite
    # Create a Boto3 S3 client
    s3_client = boto3.client("s3")

    # Retrieve the list of objects in the specified S3 folder
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)

    print(f"Downloading: {folder_name}")

    # Iterate through each object in the folder
    for obj in response["Contents"]:
        # Extract the object key (filename)
        obj_key = obj["Key"]

        # Check if the object is a file
        if obj["Size"] > 0:
            # Generate the local file path by concatenating the local directory and object key
            local_file_path = os.path.join(local_directory, obj_key)

            # Create the local directory if it doesn't exist
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            # Download the file from S3 to the local directory
            s3_client.download_file(bucket_name, obj_key, local_file_path)

            print(f"Downloaded: {local_file_path}")

    # Recursively process subdirectories
    for prefix in response.get("CommonPrefixes", []):
        subdirectory = prefix["Prefix"]
        subfolder_name = os.path.relpath(subdirectory, folder_name)
        sublocal_directory = os.path.join(local_directory, subfolder_name)
        get_file_from_aws(bucket_name, subdirectory, sublocal_directory)


def pull_s3_prefix(dst_dir, bucket, prefix):
    client = boto3.client("s3")
    resource = boto3.resource("s3")
    download_dir(client, resource, prefix, prefix, dst_dir, bucket)


def download_dir(client, resource, prefix, start_prefix, local, bucket):
    paginator = client.get_paginator("list_objects")
    for result in paginator.paginate(Bucket=bucket, Delimiter="/", Prefix=prefix):
        if result.get("CommonPrefixes") is not None:
            for subdir in result.get("CommonPrefixes"):
                download_dir(
                    client, resource, subdir.get("Prefix"), start_prefix, local, bucket
                )
        if result.get("Contents") is not None:
            for file in result.get("Contents"):
                key_relative = file.get("Key").replace(start_prefix, "")
                if not os.path.exists(os.path.dirname(local + os.sep + key_relative)):
                    os.makedirs(os.path.dirname(local + os.sep + key_relative))
                local_path = local + os.sep + key_relative
                s3_path = file.get("Key")
                print(f"Downloading {s3_path} -> {local_path}")
                resource.meta.client.download_file(bucket, s3_path, local_path)


def upload_dir_s3(source_dir, dst_bucket, dst_prefix=""):
    client = boto3.client("s3")

    # enumerate local files recursively
    for root, dirs, files in os.walk(source_dir):
        for filename in files:
            # construct the full local path
            local_path = os.path.join(root, filename)

            # construct the full Dropbox path
            relative_path = os.path.relpath(local_path, source_dir)
            s3_path = os.path.join(dst_prefix, relative_path)
            # s3_path = "{dst_prefix}{s3_path}"
            # relative_path = os.path.relpath(os.path.join(root, filename))

            try:
                print("Uploading %s..." % s3_path)
                client.upload_file(local_path, dst_bucket, s3_path)

                # try:
                # client.delete_object(Bucket=bucket, Key=s3_path)
                # except:
                # print "Unable to delete %s..." % s3_path
            except Exception as e:
                print(e)
                print(f"Failed to upload {local_path} to {s3_path}")


def set_env_variables(args):
    if args.wandb_api_key is not None:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key

    if args.aws_access_key_id is not None:
        os.environ["AWS_ACCESS_KEY_ID"] = args.aws_access_key_id

    if args.aws_secret_access_key is not None:
        os.environ["AWS_SECRET_ACCESS_KEY"] = args.aws_secret_access_key

    if args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    if args.wandb_notes is not None:
        os.environ["WANDB_NOTES"] = args.wandb_notes

    if args.hf_auth_token is not None:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = args.hf_auth_token


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])
