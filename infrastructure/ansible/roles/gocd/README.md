gocd
====

[![Ansible Role](https://img.shields.io/ansible/role/3961.svg)](https://galaxy.ansible.com/list#/roles/3961)

Installs and configures Go CD

Requirements
------------

This role requires Ansible 1.4 or higher.

Role Variables
--------------

| Name                | Default   | Description                        |
|---------------------|-----------|------------------------------------|
| gocd_server_version | 15.2.0    | Version of Go CD server to install |
| gocd_server         | false     | Install Go CD server               |
| gocd_agent_version  | 15.2.0    | Version of Go CD agent to install  |
| gocd_agent          | true      | Install Go CD agent                |
| gocd_agent_server   | 127.0.0.1 | IP address or name of Go CD server |

Dependencies
------------

- kbrebanov.java (OpenJDK 7)
- kbrebanov.unzip

Example Playbook
----------------

Install Go CD agent
```
- hosts: all
  roles:
    - { role: kbrebanov.gocd }
```

Install Go CD agent specifying server address
```
- hosts: all
  roles:
    - { role: kbrebanov.gocd, gocd_agent_server: 192.168.1.1 }
```

Install Go CD server
```
- hosts: all
  roles:
    - { role: kbrebanov.gocd, gocd_server: true, gocd_agent: false }
```

Install Go CD server & agent
```
- hosts: all
  roles:
    - { role: kbrebanov.gocd, gocd_server: true, gocd_agent: true }
```

License
-------

BSD

Author Information
------------------

Kevin Brebanov
