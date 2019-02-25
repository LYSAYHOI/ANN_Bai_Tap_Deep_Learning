# Bước 01: Nhập khẩu
nhập khẩu os
nhập ngẫu nhiên
nhập numpy như np
nhập khẩu skimage.data
nhập khẩu skimage.transform
nhập khẩu matplotlib
nhập matplotlib.pyplot dưới dạng plt
nhập hàng chục như tf
cảnh báo nhập khẩu
cảnh báo.filterwarnings ( " bỏ qua " )

# Bước 02: Đường dẫn đường bộ và hướng dẫn cách mạng và hướng dẫn
ROOT_PATH  =  " ./traffic "
train_data_dir = os.path.join ( ROOT_PATH , " bộ dữ liệu / BỉTS / Đào tạo " )
test_data_dir = os.path.join ( ROOT_PATH , " bộ dữ liệu / BỉTS / Kiểm tra " )
# in (os.path.abspath (train_data_dir))
# ----------------------------------------------
# Bước 03: Đọc / Tải dữ liệu ra ra
# Chức năng Tải một tập dữ liệu và trả về hai danh sách hình ảnh và nhãn corressponding
def  load_data ( data_dir ):
    "" " 
        Tải một tập dữ liệu và trả về hai danh sách:
           - hình ảnh: một danh sách các mảng Numpy, mỗi mảng đại diện cho một hình ảnh.
           - nhãn: danh sách các số đại diện cho nhãn hình ảnh.
    "" "
    # Nhận tất cả các thư mục con của data_dir. Mỗi đại diện cho một nhãn hiệu.
    thư mục = [d cho d trong os.listdir (data_dir)
                   if os.path.itorir (os.path.join (data_dir, d))]
    # Lặp qua các thư mục nhãn và thu thập dữ liệu trong
    # Khai báo Hai danh sách: nhãn và hình ảnh.
    nhãn = []
    hình ảnh = []
    cho d trong thư mục:
        nhãn_dir = os.path.join (data_dir, d)
        file_names = [os.path.join (nhãn_dir, f)
                      cho f trong os.listdir (nhãn_dir) nếu f.endswith ( " .ppm " )]
        # Đối với mỗi nhãn, tải hình ảnh của nó và thêm chúng vào danh sách hình ảnh.
        # Và thêm số nhãn (tức là tên thư mục) vào danh sách nhãn.
        cho f trong file_names:
            hình ảnh.append (skimage.data.imread (f))
            nhãn.append ( int (d))
    trả lại hình ảnh, nhãn

# ----------------------------------------------
# Bước 04: XỬ LÝ HÌNH ẢNH HÌNH ẢNH KHÁC NHAU ==> GIẢI QUYẾT HÌNH ẢNH
in ( " Tập dữ liệu đào tạo đang tải ...... " )
hình ảnh, nhãn = load_data (train_data_dir)
in ( " Tập dữ liệu đã được tải! " )
in ( " ----------------------------------------------- -------- " )
in ( " Trước khi thay đổi kích thước hình ảnh thành 32x32 [hình ảnh] " )
cho hình ảnh trong hình ảnh [: 5 ]:
   in ( " hình dạng: {0} , \
    tối thiểu: {1} , tối đa: {2} " \
        .format (image.shape, image.min (), image.max ()))
in ( " ----------------------------------------------- -------- " )
# Thay đổi kích thước hình ảnh thành 32x32
hình ảnh32 = [skimage.transform.resize (hình ảnh, ( 32 , 32 ), mode = ' hằng ' )
               cho hình ảnh trong hình ảnh]

def  display_images_and_labels ( hình ảnh , nhãn ):
    "" " Hiển thị hình ảnh đầu tiên của mỗi nhãn. " ""
    unique_labels =  set (nhãn)
    plt.figure ( figsize = ( 15 , 15 ))
    i =  1
    cho nhãn trong unique_labels:
        # Chọn hình ảnh đầu tiên cho mỗi nhãn.
        image = hình ảnh [nhãn.index (nhãn)]
        plt.subplot ( 8 , 8 , i)   # Một lưới gồm 8 hàng x 8 cột
        plt.axis ( ' tắt ' )
        plt.title ( " Nhãn {0} ( {1} ) " .format (nhãn, nhãn.count (nhãn)))
        tôi + =  1
        _ = plt.imshow (hình ảnh)
    plt.show ()
# display_images_and_labels (hình ảnh 32, nhãn)
#
in ( " ----------------------------------------------- -------- " )
in ( " Sau khi thay đổi kích thước hình ảnh thành 32x32 [hình ảnh 32] " )
cho hình ảnh trong hình ảnh32 [: 5 ]:
    in ( " hình dạng: {0} , \
    tối thiểu: {1} , tối đa: {2} " \
        .format (image.shape, image.min (), image.max ()))
in ( " ----------------------------------------------- -------- " )

# plt.subplot (211)
# plt.imshow (hình ảnh [0])
# plt.subplot (212)
# plt.imshow (hình ảnh 32 [0])
# plt.show ()

nhãn_a = np.array (nhãn)
hình ảnh_a = np.array (hình ảnh 32)
in ( " nhãn: " , nhãn_a.shape, " \ n hình ảnh: " , hình ảnh_a.shape)
# ------------- Mô hình khả thi tối thiểu --------------------------

# Tạo một biểu đồ để giữ mô hình.
đồ thị = tf.Graph ()
# Tạo mô hình trong biểu đồ.
với graph.as_default ():
    # Giữ chỗ cho đầu vào và nhãn.
    hình ảnh_ph = tf.placeholder (tf.float32, [ Không , 32 , 32 , 3 ])
    nhãn_ph = tf.placeholder (tf.int32, [ Không ])

    # Làm phẳng đầu vào từ: [Không, chiều cao, chiều rộng, kênh]
    # Đến lớp đầu vào: [Không có, chiều cao * chiều rộng * kênh] == [Không có, 3072]
    hình ảnh_flat = tf.contrib.layers.flatten (hình ảnh_ph)

    # 1 Lớp ẩn: 1024 đơn vị
    xx1 = tf.contrib.layers.fully_connected (hình ảnh_flat, 128 , tf.nn.relu)
    xx2 = tf.contrib.layers.fully_connected (xx1, 128 , tf.nn.relu)
    # Lớp đầu ra
    # Tạo các bản ghi kích thước [Không, 62]
    logits = tf.contrib.layers.fully_connected (xx2, 62 , tf.nn.relu)

    # Chuyển đổi nhật ký thành chỉ mục nhãn (int).
    # Hình dạng [Không], là vectơ 1D có độ dài == batch_size.
    dự đoán_labels = tf.argmax (logits, 1 )

    # Xác định hàm mất.
    # Entropy chéo là một lựa chọn tốt để phân loại.
    loss = tf.reduce_mean (tf.nn.spzzy_softmax_cross_entropy_with_logits ( logits = logits, nhãn = nhãn_ph))

    # Tạo đào tạo op.
    train = tf.train.AdamOptimizer ( learning_rate = 0,001 ). tối đa hóa (mất)

    # Và cuối cùng, một op khởi tạo để thực thi trước khi đào tạo.
    init = tf.global_variables_initializer ()

in ( "hình ảnh_flat: " , hình ảnh_flat)
# in ("lớp ẩn:", xx)
in ( "nhật ký : " , nhật ký)
in ( " mất: " , mất)
in ( " dự đoán_labels: " , dự đoán_labels)
in ( " Bat dau train " )
# ------------- ĐÀO TẠO --------------------------
# Tạo một phiên để chạy biểu đồ chúng tôi đã tạo.
session = tf.Session ( đồ thị = đồ thị)

# Bước đầu tiên là luôn luôn khởi tạo tất cả các biến.
# Tuy nhiên, chúng tôi không quan tâm đến giá trị trả lại. Không có gì.
_ = session.run ([init])
cho tôi trong  phạm vi ( 1001 ):
    _, loss_value = session.run ([tàu, mất],
                                feed_dict =
                                {hình ảnh_ph: hình ảnh_a, 
                                nhãn_ph: nhãn_a})
    nếu tôi %  100  ==  0 :
        in ( " Mất: " , loss_value)

# ------------- SỬ DỤNG MÔ HÌNH --------------------------
# Chọn 10 hình ảnh ngẫu nhiên
sample_indexes = Random.sample ( phạm vi ( len (hình ảnh 32)), 10 )
sample_images = [hình ảnh32 [i] cho i trong sample_indexes]
sample_labels = [nhãn [i] cho i trong sample_indexes]

# Chạy op "dự đoán_labels" op.
dự đoán = session.run ([dự đoán_labels],
                        feed_dict = {hình ảnh_ph: sample_images}) [ 0 ]
in (sample_labels)
in (dự đoán)

# Hiển thị dự đoán và sự thật mặt đất một cách trực quan.
fig = plt.figure ( figsize = ( 10 , 10 ))
cho i trong  phạm vi ( len (sample_images)):
    sự thật = sample_labels [i]
    dự đoán = dự đoán [i]
    plt.subplot ( 5 , 2 , 1 + i)
    plt.axis ( ' tắt ' )
    màu = ' xanh '  nếu sự thật == dự đoán khác  ' đỏ '
    plt.text ( 40 , 10 ,
             " Sự thật:         {0} \ n Dự đoán: {1} " .format (sự thật, dự đoán),
             phông chữ = 12 , màu = màu)
    plt.imshow (sample_images [i])
    
plt.show ()

# ------------- ĐÁNH GIÁ --------------------------
# Tải tập dữ liệu thử nghiệm.
test_images, test_labels = load_data (test_data_dir)
# Chuyển đổi hình ảnh, giống như chúng ta đã làm với tập huấn luyện.
test_images32 = [skimage.transform.resize (hình ảnh, ( 32 , 32 ), mode = ' hằng ' )
                 cho hình ảnh trong test_images]
# display_images_and_labels (test_images32, test_labels)
# Chạy dự đoán so với bộ thử nghiệm đầy đủ.
dự đoán = session.run ([dự đoán_labels],
                        feed_dict = {hình ảnh_ph: test_images32}) [ 0 ]
# Tính số lượng trận đấu chúng tôi có.
match_count =  sum ([ int (y == y_) cho y, y_ trong  zip (test_labels, dự đoán)])
độ chính xác = match_count /  len (test_labels)
in ( " Độ chính xác: { : .3f } " .format (độ chính xác))
# Đóng phiên. Điều này sẽ phá hủy mô hình được đào tạo.
session.c Đóng ()
