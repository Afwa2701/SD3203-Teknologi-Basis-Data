Nama : Afwa Fuadi Nugraha
NIM : 121450019
Kelas : TBD RB

## Tiga Cara Dalam Menyimpan dan Mengakses Banyak Gambar Pada Python
### Setup Dataset
Langkah-langkah untuk memuat dataset CIFAR-10 dijelaskan dengan jelas. Pertama, library yang diperlukan diimpor, termasuk `numpy` untuk manipulasi array, `pickle` untuk membaca data serial, dan `Path` dari `pathlib` untuk mengelola path file. Path menuju dataset CIFAR-10 ditetapkan dalam variabel `data_dir`. Selanjutnya, sebuah fungsi bernama `unpickle` didefinisikan untuk membaca file pickle CIFAR-10. Setelah itu, dilakukan iterasi melalui setiap file batch di direktori data. Setiap file batch dibuka menggunakan fungsi `unpickle`, dan setiap gambar dalam batch diambil sebagai array 1D yang diperluas (flatten). Array tersebut kemudian dipisahkan menjadi tiga saluran untuk merangkai kembali gambar RGB yang asli, dan seluruh gambar-gambar tersebut disusun kembali menjadi array 3D menggunakan `np.dstack`. Label dari setiap gambar juga disimpan. Akhirnya, bentuk dari array gambar dan label dicetak untuk memverifikasi jumlah dan dimensi data yang dimuat. 

```python
import numpy as np
import pickle
from pathlib import Path
# Path to the unzipped CIFAR data
data_dir = Path("D:/Downloads/Afwa/KULIAH/Semester-6/cifar-10-python/cifar-10-batches-py")
# Unpickle function provided by the CIFAR hosts
def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict
images, labels = [], []
for batch in data_dir.glob("data_batch_*"):
    batch_data = unpickle(batch)
    for i, flat_im in enumerate(batch_data[b"data"]):
        im_channels = []
        # Each image is flattened, with channels in order of R, G, B
        for j in range(3):
            im_channels.append(
                flat_im[j * 1024 : (j + 1) * 1024].reshape((32, 32))
            )
        # Reconstruct the original image
        images.append(np.dstack((im_channels)))
        # Save the label
        labels.append(batch_data[b"labels"][i])
print("Loaded CIFAR-10 training set:")
print(f" - np.shape(images)     {np.shape(images)}")
print(f" - np.shape(labels)     {np.shape(labels)}")
```
{.output .stream .stdout}
    Loaded CIFAR-10 training set:
     - np.shape(images)     (50000, 32, 32, 3)
     - np.shape(labels)     (50000,)

### Setup untuk menyimpan gambar dalam disk
Untuk mempersiapkan lingkungan untuk menyimpan dan mengakses gambar di disk, langkah-langkah berikut perlu dijalankan dalam shell perangkat yang digunakan:

```python
pip install Pillow
conda install -c conda-forge pillow
```
Dengan demikian, setelah langkah-langkah ini selesai dijalankan, lingkungan akan siap untuk menyimpan dan mengakses gambar pada disk menggunakan Python.

### Mempersiapkan LMDB
LMDB memetakan data langsung ke dalam memori, yang berarti bahwa ia mengembalikan penunjuk langsung ke alamat memori dari kunci dan nilai, tanpa perlu menyalin data ke dalam memori seperti yang biasanya dilakukan oleh sebagian besar database lainnya. Untuk menyimpan dan mengakses gambar dalam format LMDB, langkah-langkah berikut perlu dilakukan dalam shell Python pada perangkat yang digunakan:

```python
pip install lmdb
conda install -c conda-forge python-lmdb
```
Dengan melakukan langkah-langkah ini, lingkungan akan disiapkan untuk menyimpan dan mengakses gambar pada disk menggunakan LMDB.

### Mempersiapkan HDF5
HDF5 adalah singkatan dari Hierarchical Data Format. File HDF terdiri dari dua jenis objek: Dataset dan Grup

```pyhon
pip install h5py
conda install -c conda-forge h5py
```
## Storing a Single Image
Proses awal melibatkan persiapan folder tergantung pada metode yang akan digunakan, dimana setiap folder akan berisi file gambar dari database. Untuk mengilustrasikan perbandingan kinerja antara jumlah file, kami memulai dari 1 gambar hingga 100.000 gambar. Dalam kasus CIFAR-10 dengan lima set data, total gambar adalah 50.000, sehingga setiap gambar akan diproses dua kali untuk mencapai jumlah 100.000 gambar.

```python
from pathlib import Path
disk_dir = Path("data/disk/")
lmdb_dir = Path("data/lmdb/")
hdf5_dir = Path("data/hdf5/")
disk_dir.mkdir(parents=True, exist_ok=True)
lmdb_dir.mkdir(parents=True, exist_ok=True)
hdf5_dir.mkdir(parents=True, exist_ok=True)
```
### Menyimpan ke Disk
```python
from PIL import Image
import csv
def store_single_disk(image, image_id, label):
    """ Stores a single image as a .png file on disk.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    Image.fromarray(image).save(disk_dir / f"{image_id}.png")
    with open(disk_dir / f"{image_id}.csv", "wt") as csvfile:
        writer = csv.writer(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        writer.writerow([label])
```

Fungsi ini bertugas menyimpan gambar dalam format .png ke dalam disk dan label gambar ke dalam file .csv. Fungsinya menerima tiga parameter: image (array gambar dengan ukuran (32, 32, 3)), image_id (ID unik dalam bentuk integer untuk gambar tersebut), dan label (label gambar). Proses dimulai dengan mengonversi gambar menjadi objek gambar menggunakan modul PIL (Python Imaging Library), kemudian menyimpannya ke dalam disk dengan menggunakan fungsi save. Selanjutnya, label gambar disimpan ke dalam file CSV dengan label ditulis pada baris pertama dari file tersebut.

### Menyimpan ke LMBD
Kelas `CIFAR_Image` ini dirancang untuk merepresentasikan sebuah gambar CIFAR beserta labelnya. Pada inisialisasi, ia menyimpan dimensi gambar untuk rekonstruksi (meskipun tidak benar-benar diperlukan untuk dataset ini), dan kemudian mengonversi gambar ke dalam format byte menggunakan metode `tobytes()`. Selain itu, label gambar juga disimpan. Metode `get_image()` digunakan untuk mengembalikan gambar sebagai numpy array setelah mengonversi kembali dari format byte.

```python
class CIFAR_Image:
    def __init__(self, image, label):
        # Dimensions of image for reconstruction - not really necessary
        # for this dataset, but some datasets may include images of
        # varying sizes
        self.channels = image.shape[2]
        self.size = image.shape[:2]
        self.image = image.tobytes()
        self.label = label
    def get_image(self):
        """ Returns the image as a numpy array. """
        image = np.frombuffer(self.image, dtype=np.uint8)
        return image.reshape(*self.size, self.channels)
```

Fungsi `store_single_lmdb` digunakan untuk menyimpan sebuah gambar CIFAR ke dalam database LMDB. Dalam proses ini, gambar direpresentasikan sebagai array numpy dengan ukuran (32, 32, 3). Fungsi ini menerima tiga parameter: `image` (array gambar), `image_id` (ID unik gambar dalam bentuk integer), dan `label` (label gambar).

Pertama, fungsi menghitung ukuran pemetaan (map size) yang diperlukan untuk database LMDB berdasarkan ukuran gambar. Kemudian, sebuah lingkungan LMDB baru dibuat menggunakan `lmdb.open`, dengan ukuran pemetaan yang telah dihitung sebelumnya.

Setelah itu, transaksi tulis baru dimulai menggunakan `env.begin(write=True)`. Semua pasangan kunci-nilai dalam LMDB harus berupa string, oleh karena itu, objek `CIFAR_Image` dibuat untuk merepresentasikan gambar dan labelnya. Key disusun dalam format string menggunakan ID gambar, kemudian pasangan key-value tersebut disimpan dalam transaksi menggunakan `txn.put`.

Setelah transaksi selesai, lingkungan LMDB ditutup.

```python
import lmdb
import pickle
def store_single_lmdb(image, image_id, label):
    """ Stores a single image to a LMDB.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    map_size = image.nbytes * 10
    # Create a new LMDB environment
    env = lmdb.open(str(lmdb_dir / f"single_lmdb"), map_size=map_size)
    # Start a new write transaction
    with env.begin(write=True) as txn:
        # All key-value pairs need to be strings
        value = CIFAR_Image(image, label)
        key = f"{image_id:08}"
        txn.put(key.encode("ascii"), pickle.dumps(value))
    env.close()
```

### Menyimpan ke HDF5
```python
import h5py
def store_single_hdf5(image, image_id, label):
    """ Stores a single image to an HDF5 file.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    # Create a new HDF5 file
    file = h5py.File(hdf5_dir / f"{image_id}.h5", "w")
    # Create a dataset in the file
    dataset = file.create_dataset(
        "image", np.shape(image), h5py.h5t.STD_U8BE, data=image
    )
    meta_set = file.create_dataset(
        "meta", np.shape(label), h5py.h5t.STD_U8BE, data=label
    )
    file.close()
```
Fungsi `store_single_hdf5` dirancang untuk menyimpan sebuah gambar CIFAR ke dalam sebuah file HDF5. Dengan menerima tiga parameter: `image`, `image_id`, dan `label`, fungsi ini mengawali proses dengan membuat file HDF5 baru menggunakan `h5py.File`, menggunakan ID gambar sebagai nama file. Kemudian, array gambar disimpan dalam sebuah dataset dengan nama "image", yang menggunakan tipe data unsigned integer 8-bit. Selain itu, dataset meta dibuat untuk menyimpan label gambar. Setelah dataset dibuat, file HDF5 ditutup dengan `file.close()`.

### Eksperimen Storing Single Images
```python
_store_single_funcs = dict(
    disk=store_single_disk, lmdb=store_single_lmdb, hdf5=store_single_hdf5
)
```


```python
from timeit import timeit
store_single_timings = dict()
for method in ("disk", "lmdb", "hdf5"):
    t = timeit(
        "_store_single_funcs[method](image, 0, label)",
        setup="image=images[0]; label=labels[0]",
        number=1,
        globals=globals(),
    )
    store_single_timings[method] = t
    print(f"Method: {method}, Time usage: {t}")
```

{Output}
Method: disk, Time usage: 0.4261889010325463
Method: lmdb, Time usage: 0.5620705989710595
Method: hdf5, Time usage: 0.28305170011717033

Pada kode di atas, sebuah kamus `_store_single_funcs` dibuat yang berisi fungsi-fungsi untuk menyimpan gambar dalam format disk, LMDB, dan HDF5. Selanjutnya, dilakukan pengukuran waktu eksekusi untuk setiap metode penyimpanan gambar menggunakan fungsi `timeit`. Melalui iterasi untuk setiap metode ("disk", "lmdb", "hdf5"), waktu eksekusi untuk menyimpan satu gambar diukur. Hasilnya kemudian dicatat dalam kamus `store_single_timings`, yang menunjukkan waktu eksekusi untuk setiap metode. Hasil pengukuran waktu tersebut kemudian dicetak dalam format "Method: <method>, Time usage: <time>".

## **Storing Many Images**

### Menyesuaikan Kode Untuk Many Images
Tiga fungsi baru telah dibuat untuk menyimpan beberapa gambar dalam berkas bertipe .png, LMDB, dan HDF5. Fungsi `store_many_disk` menyimpan array gambar ke dalam disk, disertai dengan label yang disimpan dalam file CSV terpisah. Fungsi `store_many_lmdb` menyimpan seluruh array gambar ke dalam sebuah basis data LMDB dalam satu transaksi tulis. Sementara itu, fungsi `store_many_hdf5` digunakan untuk menyimpan array gambar ke dalam sebuah berkas HDF5, dengan dataset yang berisi array gambar dan dataset lain untuk array label.

```python
store_many_disk(images, labels):
    """ Stores an array of images to disk
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)
    # Save all the images one by one
    for i, image in enumerate(images):
        Image.fromarray(image).save(disk_dir / f"{i}.png")
    # Save all the labels to the csv file
    with open(disk_dir / f"{num_images}.csv", "w") as csvfile:
        writer = csv.writer(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        for label in labels:
            # This typically would be more than just one value per row
            writer.writerow([label])
def store_many_lmdb(images, labels):
    """ Stores an array of images to LMDB.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)
    map_size = num_images * images[0].nbytes * 10
    # Create a new LMDB DB for all the images
    env = lmdb.open(str(lmdb_dir / f"{num_images}_lmdb"), map_size=map_size)
    # Same as before — but let's write all the images in a single transaction
    with env.begin(write=True) as txn:
        for i in range(num_images):
            # All key-value pairs need to be Strings
            value = CIFAR_Image(images[i], labels[i])
            key = f"{i:08}"
            txn.put(key.encode("ascii"), pickle.dumps(value))
    env.close()
def store_many_hdf5(images, labels):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)
    # Create a new HDF5 file
    file = h5py.File(hdf5_dir / f"{num_images}_many.h5", "w")
    # Create a dataset in the file
    dataset = file.create_dataset(
        "images", np.shape(images), h5py.h5t.STD_U8BE, data=images
    )
    meta_set = file.create_dataset(
        "meta", np.shape(labels), h5py.h5t.STD_U8BE, data=labels
    )
    file.close()
```

### Mempersiapkan Dataset
```python
cutoffs = [10, 100, 1000, 10000, 100000]
# Let's double our images so that we have 100,000
images = np.concatenate((images, images), axis=0)
labels = np.concatenate((labels, labels), axis=0)
# Make sure you actually have 100,000 images and labels
print(np.shape(images))
print(np.shape(labels))
```
::: {.output .stream .stdout}
    (100000, 32, 32, 3)
    (100000,)

Terdapat penambahan kode yang melakukan penggandaan array gambar dan label sehingga jumlahnya menjadi 100.000. Hasil cetak menunjukkan ukuran array gambar sekarang adalah (100.000, 32, 32, 3) dan ukuran array label adalah (100.000,).

### Eksperimen Storing Many Images
Dalam implementasi, fungsi-fungsi untuk menyimpan array gambar dan label menggunakan tiga metode yang berbeda (disk, lmdb, dan hdf5) disimpan dalam dictionary `_store_many_funcs`. Dilakukan iterasi melalui nilai-nilai dalam variabel `cutoffs`, di mana setiap metode penyimpanan dievaluasi. Waktu eksekusi diukur menggunakan `timeit`, dan hasilnya dicatat ke dalam dictionary `store_many_timings`.

```python
_store_many_funcs = dict(
    disk=store_many_disk, lmdb=store_many_lmdb, hdf5=store_many_hdf5
)
from timeit import timeit
store_many_timings = {"disk": [], "lmdb": [], "hdf5": []}
for cutoff in cutoffs:
    for method in ("disk", "lmdb", "hdf5"):
        t = timeit(
            "_store_many_funcs[method](images_, labels_)",
            setup="images_=images[:cutoff]; labels_=labels[:cutoff]",
            number=1,
            globals=globals(),
        )
        store_many_timings[method].append(t)
        # Print out the method, cutoff, and elapsed time
        print(f"Method: {method}, Time usage: {t}")
```


## **Reading a Single Image**

### Membaca dari Disk

```python
def read_single_disk(image_id):
    """ Stores a single image to disk.
        Parameters:
        ---------------
        image_id    integer unique ID for image
        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
    """
    image = np.array(Image.open(disk_dir / f"{image_id}.png"))
    with open(disk_dir / f"{image_id}.csv", "r") as csvfile:
        reader = csv.reader(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        label = int(next(reader)[0])
    return image, label
```
Fungsi `read_single_disk` digunakan untuk membaca sebuah gambar dan label terkait dari disk. Saat dipanggil dengan parameter `image_id`, fungsi ini membuka gambar dengan nama file yang sesuai dari direktori disk dan membacanya sebagai array gambar menggunakan modul PIL. Selain itu, fungsi ini membuka file CSV yang menyimpan label gambar dan mengambil label tersebut sebagai integer. Hasilnya adalah pasangan gambar dan label yang dikembalikan oleh fungsi ini.

### Membaca dari LMDB
```python
def read_single_lmdb(image_id):
    """ Stores a single image to LMDB.
        Parameters:
        ---------------
        image_id    integer unique ID for image
        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
    """
    # Open the LMDB environment
    env = lmdb.open(str(lmdb_dir / f"single_lmdb"), readonly=True)
    # Start a new read transaction
    with env.begin() as txn:
        # Encode the key the same way as we stored it
        data = txn.get(f"{image_id:08}".encode("ascii"))
        # Remember it's a CIFAR_Image object that is loaded
        cifar_image = pickle.loads(data)
        # Retrieve the relevant bits
        image = cifar_image.get_image()
        label = cifar_image.label
    env.close()
    return image, label
```
Fungsi `read_single_lmdb` digunakan untuk membaca sebuah gambar dan label terkait dari sebuah basis data LMDB. Ketika dipanggil dengan parameter `image_id`, fungsi ini membuka lingkungan LMDB dengan mode baca saja dan memulai transaksi baca baru. Kemudian, fungsi ini mengambil data yang terkait dengan kunci yang sesuai dengan `image_id` dari transaksi tersebut, dan mengembalikan gambar dan label yang terkait setelah mendekode data tersebut menggunakan modul pickle.


### Membaca dari HDF5
Fungsi `read_single_hdf5` membaca gambar dan label dari sebuah berkas HDF5 dengan menggunakan modul h5py. Dengan menerima parameter `image_id`, fungsi ini membuka berkas HDF5 dalam mode baca dan tulis ("r+"). Gambar dan label kemudian diambil dari dataset yang sesuai dalam berkas tersebut, dengan dataset gambar diambil dari path "/image" dan dataset label dari path "/meta". Data dari kedua dataset tersebut diubah menjadi array numpy dengan tipe data "uint8" untuk memastikan konsistensi tipe data. Hasilnya, fungsi mengembalikan gambar dan label dalam bentuk tuple, di mana gambar direpresentasikan sebagai array numpy 32x32x3 dan label sebagai integer.

```python
def read_single_hdf5(image_id):
    """ Stores a single image to HDF5.
        Parameters:
        ---------------
        image_id    integer unique ID for image
        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
    """
    # Open the HDF5 file
    file = h5py.File(hdf5_dir / f"{image_id}.h5", "r+")
    image = np.array(file["/image"]).astype("uint8")
    label = int(np.array(file["/meta"]).astype("uint8"))
    return image, label
```

Kode dibawah menetapkan fungsi-fungsi untuk membaca gambar dari tiga jenis penyimpanan yang berbeda ke dalam sebuah kamus yang disebut `_read_single_funcs`. Tiga jenis penyimpanan tersebut adalah disk, lmdb, dan hdf5, dan masing-masing fungsi pembacaan diberi nama `read_single_disk`, `read_single_lmdb`, dan `read_single_hdf5`. Dengan menyimpan fungsi-fungsi ini dalam kamus, memungkinkan penggunaan fleksibel tergantung pada jenis penyimpanan yang digunakan untuk menyimpan gambar. Kamus ini kemungkinan akan digunakan dalam konteks aplikasi yang lebih besar untuk mengelola proses pembacaan gambar dari penyimpanan yang berbeda.

```python
_read_single_funcs = dict(
    disk=read_single_disk, lmdb=read_single_lmdb, hdf5=read_single_hdf5
)
```

### Eksperimen Reading a Single Image

```python
from timeit import timeit
read_single_timings = dict()
for method in ("disk", "lmdb", "hdf5"):
    t = timeit(
        "_read_single_funcs[method](0)",
        setup="image=images[0]; label=labels[0]",
        number=1,
        globals=globals(),
    )
    read_single_timings[method] = t
    print(f"Method: {method}, Time usage: {t}")
```
Menghitung waktu yang diperlukan untuk membaca sebuah gambar menggunakan tiga metode penyimpanan yang berbeda: disk, lmdb, dan hdf5. Iterasi dilakukan melalui setiap metode, di mana setiap metode dijalankan sekali dengan menggunakan fungsi `timeit` untuk mengukur waktu eksekusinya. Hasil waktu eksekusi kemudian dicatat dalam kamus `read_single_timings`, dan hasilnya dicetak untuk setiap metode. Proses ini memberikan pemahaman tentang performa relatif dari setiap metode dalam membaca gambar dari penyimpanan yang berbeda.


## **Reading Many Images**

### Menyesuaikan Kode Untuk Many Images
```python
def read_many_disk(num_images):
    """ Reads image from disk.
        Parameters:
        ---------------
        num_images   number of images to read
        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []
    # Loop over all IDs and read each image in one by one
    for image_id in range(num_images):
        images.append(np.array(Image.open(disk_dir / f"{image_id}.png")))
    with open(disk_dir / f"{num_images}.csv", "r") as csvfile:
        reader = csv.reader(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        for row in reader:
            labels.append(int(row[0]))
    return images, labels
def read_many_lmdb(num_images):
    """ Reads image from LMDB.
        Parameters:
        ---------------
        num_images   number of images to read
        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []
    env = lmdb.open(str(lmdb_dir / f"{num_images}_lmdb"), readonly=True)
    # Start a new read transaction
    with env.begin() as txn:
        # Read all images in one single transaction, with one lock
        # We could split this up into multiple transactions if needed
        for image_id in range(num_images):
            data = txn.get(f"{image_id:08}".encode("ascii"))
            # Remember that it's a CIFAR_Image object
            # that is stored as the value
            cifar_image = pickle.loads(data)
            # Retrieve the relevant bits
            images.append(cifar_image.get_image())
            labels.append(cifar_image.label)
    env.close()
    return images, labels
def read_many_hdf5(num_images):
    """ Reads image from HDF5.
        Parameters:
        ---------------
        num_images   number of images to read
        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []
    # Open the HDF5 file
    file = h5py.File(hdf5_dir / f"{num_images}_many.h5", "r+")
    images = np.array(file["/images"]).astype("uint8")
    labels = np.array(file["/meta"]).astype("uint8")
    return images, labels
_read_many_funcs = dict(
    disk=read_many_disk, lmdb=read_many_lmdb, hdf5=read_many_hdf5
)
```
Tiga fungsi baru telah ditambahkan untuk membaca banyak gambar dari penyimpanan dengan tiga metode yang berbeda: disk, lmdb, dan hdf5. Pertama, fungsi `read_many_disk` membaca gambar dari disk dengan membuka setiap gambar menggunakan `Image.open()` dari modul PIL dan membacanya ke dalam array numpy. Label-label gambar dibaca dari file CSV yang sesuai. Fungsi `read_many_lmdb` membaca gambar dari basis data LMDB dengan membuka lingkungan LMDB dan membaca setiap gambar dalam satu transaksi baca. Fungsi `read_many_hdf5` membaca gambar dari berkas HDF5 dengan membuka berkas tersebut dan membaca dataset gambar dan label dari path yang sesuai di dalam berkas HDF5. Setelah dataset dibaca, gambar dan label dikembalikan dalam bentuk array numpy. Semua fungsi ini menerima satu parameter `num_images` yang menentukan jumlah gambar yang akan dibaca, dan mengembalikan array gambar dan array label. Dalam implementasi ini, fungsi-fungsi ini kemudian dimasukkan ke dalam dictionary `_read_many_funcs` untuk digunakan dalam pengukuran waktu eksekusi.

### Eksperimen Reading Many Images
```python
from timeit import timeit
read_many_timings = {"disk": [], "lmdb": [], "hdf5": []}
for cutoff in cutoffs:
    for method in ("disk", "lmdb", "hdf5"):
        t = timeit(
            "_read_many_funcs[method](num_images)",
            setup="num_images=cutoff",
            number=1,
            globals=globals(),
        )
        read_many_timings[method].append(t)
        # Print out the method, cutoff, and elapsed time
        print(f"Method: {method}, No. images: {cutoff}, Time usage: {t}")
```
Kode di atas bertujuan untuk mengukur waktu yang dibutuhkan untuk membaca banyak gambar dari setiap metode penyimpanan (disk, lmdb, hdf5) untuk berbagai jumlah gambar yang ditentukan oleh `cutoffs`. Proses ini menggunakan fungsi `timeit` untuk setiap kombinasi jumlah gambar dan metode penyimpanan. Hasil waktu eksekusi kemudian dicatat dalam dictionary `read_many_timings` sesuai dengan metode penyimpanan yang digunakan. Setiap waktu eksekusi dicetak untuk setiap iterasi untuk menampilkan metode, jumlah gambar, dan waktu yang digunakan.

## Kesimpulan
Dalam eksperimen ini, saya membandingkan kinerja dan penggunaan disk dari tiga metode penyimpanan gambar: disk, LMDB, dan HDF5. saya menyimpan dan membaca gambar CIFAR-10 menggunakan setiap metode untuk memahami pengaruhnya terhadap waktu eksekusi dan penggunaan disk. Hasilnya menunjukkan bahwa penggunaan disk biasa memiliki kinerja yang baik tetapi memerlukan ruang penyimpanan yang besar, sementara LMDB dan HDF5 membutuhkan lebih sedikit ruang disk tetapi dengan kinerja yang sedikit lebih lambat. LMDB, meskipun efisien dalam penggunaan ruang disk, memiliki kinerja yang sedikit lebih lambat dibandingkan dengan HDF5 dalam beberapa kasus. Namun, pilihan antara ketiga metode ini harus dipertimbangkan berdasarkan kebutuhan aplikasi spesifik dan faktor lain seperti efisiensi ruang penyimpanan dan kinerja keseluruhan.
