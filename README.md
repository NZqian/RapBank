---
license: cc-by-sa-4.0
task_categories:
- text-to-speech
pretty_name: RapBank
size_categories:
- 10M<n<100M
configs:
- config_name: default
  data_files:
  - split: train
    path: "rap_songs.csv"
---

# Dataset Card for RapBank

<!-- Provide a quick summary of the dataset. -->

RapBank is the first dataset for rap generation. The rap songs are collected from YouTube, and we provide a meticulously designed pipeline for data processing

## Dataset Details

### Dataset Sources [optional]

<!-- Provide the basic links for the dataset. -->

- **Repository:** https://github.com/NZqian/RapBank
- **Paper:** https://arxiv.org/abs/2408.15474
- **Demo:** https://nzqian.github.io/Freestyler/

### Statistics
The RapBank dataset comprises links to a total of 94, 164 songs.
However, due to the unavailability of certain videos, we successfully downloaded 92,371 songs, amounting to 5,586 hours of content, with an average duration of 218 seconds per song. 
These songs span 84 different languages. English has the highest duration, totaling 3,830 hours, which constitutes approximately two-thirds of the overall duration.

|       Subset      | DNSMOS Threshold | PPS Threshold | Primary Singer Threshold | Total Duration (h) | Average Segment Duration (s) |
|:-----------------:|:----------------:|:-------------:|:------------------------:|:------------------:|:----------------------------:|
|     Orig Songs    |         -        |       -       |             -            |       5586.2       |             227.7            |
|      RapBank      |         -        |       -       |             -            |       4353.6       |             17.4             |
| RapBank (English) |         -        |       -       |             -            |       3830.1       |             17.3             |
|       Basic       |        2.5       |     12-35     |            0.8           |       1322.0       |             18.5             |
|      Standard     |        3.5       |     16-32     |            0.9           |        295.3       |             18.8             |
|      Premium      |        3.8       |     18-30     |            1.0           |        58.3        |             18.7             |

### Dataset Structure

The metadata is stored in a CSV file and includes the following fields: 
```
video_id, video_title, playlist_id, playlist_title, playlist_index
```
Users can access the corresponding rap video at https://www.youtube.com/watch?v=video_id, and then use the provided pipeline for data processing.

## Processing Pipeline

The data processing pipeline includes steps such as source seperation, segmentation, lyrics recognition, the details can be found in our paper.

### Install Dependency

```
pip install -r requirements.txt
```

### Data Processing
After downloading the rap songs, place them in a `wav` folder, such as `/path/to/your/data/wav`, then simply use `pipeline.sh` to process the data
```
bash pipeline.sh /path/to/your/data /path/to/save/features start_stage stop_stage
```
the stage is ranged from 0 to 5.

Multiple GPU is recommended for faster processing speed.

## License:
The data falls under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license. More details on this license can be viewed at https://creativecommons.org/licenses/by-nc-sa/4.0/.

By downloading or using this dataset, you agree to abide by the terms of the CC BY-NC-SA 4.0 license, including any modifications or redistributions in any form.

## Disclaimer:
THIS DATASET IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


## Terms of Access
The RapBank dataset, derived from the public availiable YouTube videos, is available for download for non-commercial purposes under a The data falls under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license. More details on this license can be viewed at https://creativecommons.org/licenses/by-nc-sa/4.0/.
We do not own the copyright of the audios: the copyright remains with the original owners of the video or audio, and the public URL is provided in RapBank for the original video or audio.

Terms of Access: The Researcher has requested permission to use the RapBank database. In exchange for such permission, Researcher hereby agrees to the following terms and conditions:

1. Researcher shall use the Database only for non-commercial research and educational purposes.

2. The authors make no representations or warranties regarding the Database, including but not limited to warranties of non-infringement or fitness for a particular purpose.

3. Researcher accepts full responsibility for his or her use of the Database and shall defend and indemnify the authors of RapBank, including their employees, Trustees, officers and agents, against any and all claims arising from Researcher's use of the Database, including but not limited to Researcher's use of any copies of copyrighted audio files that he or she may create from the Database.

4. Researcher may provide research associates and colleagues with access to the Database provided that they first agree to be bound by these terms and conditions.

5. The authors reserve the right to terminate Researcher's access to the Database at any time.

6. If Researcher is employed by a for-profit, commercial entity, Researcher's employer shall also be bound by these terms and conditions, and Researcher hereby represents that he or she is fully authorized to enter into this agreement on behalf of such employer.