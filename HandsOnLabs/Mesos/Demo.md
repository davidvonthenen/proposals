- Marathon - ec2-52-22-52-161.compute-1.amazonaws.com:8080
- Master1 - ec2-52-70-118-25.compute-1.amazonaws.com:5050
- Master2 - ec2-52-22-52-161.compute-1.amazonaws.com:5050
- Master3 - ec2-52-70-114-203.compute-1.amazonaws.com:5050

- Slave1 - ec2-52-22-83-11.compute-1.amazonaws.com
- Slave2 - ec2-52-91-90-220.compute-1.amazonaws.com
- Slave3 - ec2-52-70-118-36.compute-1.amazonaws.com
- Slave4 - ec2-54-84-89-137.compute-1.amazonaws.com
- Slave5 - ec2-54-86-165-3.compute-1.amazonaws.com
- Slave6 - ec2-54-84-240-78.compute-1.amazonaws.com


- Deployments tab
- About (Current master, Marathon info, etc)
- List of Apps tab
- DEMO/Walk-Through
  - Deploy an App  
  ```
  echo 'My name is XXXX' > readme && python -m SimpleHTTPServer $PORT
  ```
  - Click the app for info - what Slave/Agent is it running on
  - Suspend the App (which moves it to 0 instances)
  - Scale up and application
  - Destroy
- REST API
  - Start the App using the REST API
    ```
    JSON:
    {
      "id": "simplewebapp",
      "cmd": "python -m SimpleHTTPServer $PORT",
      "mem": 32,
      "cpus": 0.1,
      "instances": 1
    }

    Call REST API:
    curl -i -H 'Content-Type: application/json' -d @demo.json ec2-52-22-52-161.compute-1.amazonaws.com:8080/v2/apps
    ```
  - Kill the App to witness failover
    ```
    ps auxfw | grep executor

    kill <pid>
    ```
