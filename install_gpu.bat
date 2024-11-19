rem Step 1: Activate Anaconda
call "%userprofile%\anaconda3\Scripts\activate.bat" "%userprofile%\anaconda3"

rem Step 2: Create a new environment
call conda create --name finding_a_place_gpu python -y

rem Step 3: Activate the new environment
call conda activate finding_a_place_gpu

rem Step 4: Install the required packages
call conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y

pip install -r requirements00.txt
pip install -r requirements01.txt

rem Step 5: Pause the script
pause
cmd /k