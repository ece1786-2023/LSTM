# LSTM

# git commands
git init	#initialize

#git clone https://github.com/ece1786-2023/LSTM

git fetch origin #This command fetches the latest main branch from the repo

git reset --hard origin/main #This command will set your local branch to exactly match the state of the **fetched** remote branch. **Be cautious when using --hard as it discards all local changes.**

git pull origin main #pulls new commits from github

git status	#check new/modified/untracked files

git add <file1> <file2> #**manually add all the files that were changed for each commit** 

git commit -m "msg" #**check red items in status before commit**

git remote add origin https://github.com/ece1786-2023/LSTM

git branch	#check your current branch

git branch -M branchname #creates new branch

git push -u origin branchname #push to new branch. **do not push to main!!!** create pull request and merge after pushing to new branch


# version control 
branch naming: NMMDD_HHMM_name

N=name {b=backup c=colin t=tianze}

# repo structure
