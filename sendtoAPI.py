import requests as r
import json
import base64

# Set up this API Key for GitHub

class GithubAPI(object):
    def __init__(self, oauth_token):
        """
            Define the object with token for access to Github
            
            param: 
                oauth_token: string representing for token access to Github
        """
        self.oauth_token = oauth_token
        self.headers = {'Authorization': 'token ' + self.oauth_token}
        
    def get_repo(self):
        """
            Get all repositories from Github
        """
        base_url = "https://api.github.com/user/repos"
        response = r.get(base_url, headers = self.headers)
        return response.json()
            
    def get_user_info(self):
        """
            Get information about the user
        """
        base_url = "https://api.github.com/user"
        response = r.get(base_url, headers = self.headers)
        return response.json()
        
    def get_repo_info(self, repo_name):
        """
            Get information about the repository
            
            param:
                repo_name: string representing the name of the repository
        """
        login = self.get_user_info()['login']
        base_url = "https://api.github.com/repos/{}/{}".format(login,repo_name)
        response = r.get(base_url, headers = self.headers)
        return response.json()
    
    def create_new_repo(self, name_repo, private = False):
        """
            Create a new repository
            
            pram:
                name_repo: string representing the name of the repository
                private: True if the repository is private, False otherwise
        """
        baseurl = "https://api.github.com/user/repos"
        repo_config = json.dumps({
            "name": name_repo,
            "auto_init": True,
            "private": private,
            "gitignore_template": ""
        })
        response = r.post(baseurl, data = repo_config, headers = self.headers)
        
        if (response.status_code == 201):
            return "Repository created successfully"
        if (response.status_code == 400):
            return "Bad request"
        if (response.status_code == 401):
            return "Unauthorized"
        if (response.status_code == 403):
            return "Forbidden"
        if (response.status_code == 404):
            return "Not Found"
        if (response.status_code == 422):
            return "Validation error"
        
    def create_new_file_repo(self, name_repo, name_file, data ,message_repo = "create new file"):
        """
            Create a new file in a repository
            
            param:
                name_repo: string representing the name of the repository
                name_file: string representing the name of the file
                data: string representing the content of the file
                message_repo: string representing the message of the commit
        """
        login = self.get_user_info()['login']
        base_url = "https://api.github.com/repos/{}/{}/contents/{}".format(login,name_repo,name_file)
        base64_string = base64.b64encode(bytes(data, 'utf-8'))
        response = r.put(base_url
                        , data = json.dumps({
                             "message": message_repo,
                             "content": base64_string.decode('utf-8')
                        })
                        , headers = self.headers)
        if (response.status_code == 201):
            return "created new file on repo successfully"
        if (response.status_code == 400):
            return "Bad request"
        if (response.status_code == 401):
            return "Unauthorized"
        if (response.status_code == 403):
            return "Forbidden"
        if (response.status_code == 404):
            return "Not Found"
        if (response.status_code == 422):
            return "Validation error"
        
    def get_last_repo_commit(self, name_repo):
        """
            Get the last commit of the repository on main branch
            
            param:
                name_repo: string representing the name of the repository
        """
        login = self.get_user_info()['login']
        base_url = "https://api.github.com/repos/{}/{}/commits".format(login,name_repo)
        response = r.get(base_url, headers = self.headers)
        return response.json()[0]["sha"]
    
    def get_content_new_file_on_repo(self, name_repo, name_file):
        """
            Get content of the new file on repository (force to get SHA: get hash array)
            
            param:
                name_repo: string representing the name of the repository
                name_file: string representing the name of the file
        """
        login = self.get_user_info()['login']
        base_url = "https://api.github.com/repos/{}/{}/contents/{}".format(login,name_repo,name_file)
        response = r.get(base_url, headers = self.headers)
        return response.json()['sha']
        
    def update_file_on_repo(self, name_repo, name_file, data ,message_repo = "Update new file"):
        """
            Update a file on repository
            
            param:
                name_repo: string representing the name of the repository
                name_file: string representing the name of the file
                data: string representing the content of the file
                message_repo: string representing the message of the commit
        """
        login = self.get_user_info()['login']
        base_url = "https://api.github.com/repos/{}/{}/contents/{}".format(login,name_repo,name_file)
        base64_string = base64.b64encode(bytes(data, 'utf-8'))
        response = r.put(base_url,
                         data = json.dumps({
                             "message": message_repo,
                             "content": base64_string.decode('utf-8'),
                             "sha": self.get_content_new_file_on_repo(name_repo, name_file)
                         }),
                         headers = self.headers)
        if (response.status_code == 200):
            return "Repository update successfully"
        if (response.status_code == 400):
            return "Bad request"
        if (response.status_code == 401):
            return "Unauthorized"
        if (response.status_code == 403):
            return "Forbidden"
        if (response.status_code == 404):
            return "Not Found"
        if (response.status_code == 409):
            return "Conflict"
        if (response.status_code == 422):
            return "Validation error"
        
    def get_details_file_repo(self, name_repo, name_file):
        """
            Get details of a file on repository
            
            param:
                name_repo: string representing the name of the repository
                name_file: string representing the name of the file
        """
        login = self.get_user_info()['login']
        base_url = "https://api.github.com/repos/{}/{}/contents/{}".format(login,name_repo,name_file)
        response = r.get(base_url, headers = self.headers)
        return base64.b64decode(response.json()['content']).decode('utf-8')
    
    def delete_file_on_repo(self, name_repo, name_file, message_repo = "Delete new file"):
        """
            Delete a file on repository
            
            param:
                name_repo: string representing the name of the repository
                name_file: string representing the name of the file
                message_repo: string representing the message of the commit
                
        """
        login = self.get_user_info()['login']
        base_url = "https://api.github.com/repos/{}/{}/contents/{}".format(login,name_repo,name_file)
        response = r.delete(base_url,
                            data = json.dumps({
                                "message": message_repo,
                                "sha": self.get_content_new_file_on_repo(name_repo, name_file)
                            }),
                            headers = self.headers)
        if (response.status_code == 200):
            return "File deleted successfully"
        if (response.status_code == 400):
            return "Bad request"
        if (response.status_code == 401):
            return "Unauthorized"
        if (response.status_code == 403):
            return "Forbidden"
        if (response.status_code == 404):
            return "Not Found"
        if (response.status_code == 422):
            return "Validation error"