import subprocess
import os
from datetime import datetime
from dotenv import load_dotenv

# Docker and Neo4j details
DOCKER_CONTAINER_NAME = "neo4j-apoc"  # Your Neo4j container name

# Load environment variables
load_dotenv(os.getenv('BASE_DIR')+'/env.sh')

DATA_DIR = os.getenv('DATA_DIR')
BACKUP_DIR = DATA_DIR + "/backup"  # Replace with your desired backup directory on the host

def create_backup():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"neo4j_backup_{timestamp}.dump"
    backup_path = os.path.join(BACKUP_DIR, backup_file)
    
    # Ensure the backup directory exists
    os.makedirs(BACKUP_DIR, exist_ok=True)

    # Stop the Neo4j service
    # subprocess.run(["docker", "exec", DOCKER_CONTAINER_NAME, "neo4j", "stop"], check=True)
    
    # Create a backup folder 
    subprocess.run(["docker", "exec", DOCKER_CONTAINER_NAME, "mkdir", "-p", "/backups"], check=True)
    
    # Command to create a dump inside the Docker container
    docker_command = [
        "docker", "exec", DOCKER_CONTAINER_NAME,
        "neo4j-admin", "database", "dump", "neo4j", "--to-path=/backups"
    ]

    try:
        # Run the dump command
        subprocess.run(docker_command, check=True)
        
        # Copy the dump file from the container to the host
        copy_command = [
            "docker", "cp", 
            f"{DOCKER_CONTAINER_NAME}:/backups/neo4j.dump", 
            backup_path
        ]
        subprocess.run(copy_command, check=True)
        
        print(f"Backup created successfully: {backup_path}")
        return backup_path
    except subprocess.CalledProcessError as e:
        print(f"Error creating backup: {e}")
        return None

def restore_backup(backup_file):
    # Copy the backup file to the container
    copy_command = [
        "docker", "cp",
        backup_file,
        f"{DOCKER_CONTAINER_NAME}:/backups/neo4j.dump"
    ]
    
    # Command to restore from the dump inside the Docker container
    docker_command = [
        "docker", "exec", DOCKER_CONTAINER_NAME,
        "neo4j-admin", "database", "load", "neo4j", "--from-path=/backups", "--overwrite-destination=true"
    ]

    try:
        # Copy the file to the container
        subprocess.run(copy_command, check=True)
        
        # Stop the Neo4j service
        subprocess.run(["docker", "exec", DOCKER_CONTAINER_NAME, "neo4j", "stop"], check=True)
        
        # Run the restore command
        subprocess.run(docker_command, check=True)
        
        # Start the Neo4j service
        subprocess.run(["docker", "exec", DOCKER_CONTAINER_NAME, "neo4j", "start"], check=True)
        
        print(f"Backup restored successfully from: {backup_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error restoring backup: {e}")

if __name__ == "__main__":
    backup_file = create_backup()
    if backup_file:
        print(f"Backup created at: {backup_file}")
    else:
        print("Backup creation failed.")

    # Uncomment the following line to test restore functionality
    # restore_backup(backup_file)