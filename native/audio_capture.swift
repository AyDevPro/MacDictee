// macdictee-audio : capture le son du Mac (visios, vidéos… tout sauf MacDictée) avec ScreenCaptureKit
// et l'écrit sur stdout en PCM float32 mono 16 kHz. MacDictée le lance pendant une réunion et le lit
// en continu ; fermer stdin (ou quitter MacDictée) arrête la capture.
//
// Nécessite l'autorisation « Enregistrement de l'écran et audio système » (attribuée à MacDictée).
// Compilation : swiftc -O -o macdictee-audio audio_capture.swift (voir build.sh / install.sh)
import CoreMedia
import Foundation
import ScreenCaptureKit

let stderr = FileHandle.standardError
func log(_ message: String) { stderr.write((message + "\n").data(using: .utf8)!) }

final class Capture: NSObject, SCStreamOutput, SCStreamDelegate {
    private var stream: SCStream?
    private let out = FileHandle.standardOutput
    private var announced = false

    func start() async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
        guard let display = content.displays.first else { throw NSError(domain: "macdictee", code: 1) }
        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.sampleRate = 16000
        config.channelCount = 1
        config.excludesCurrentProcessAudio = true
        // ScreenCaptureKit capture toujours un flux vidéo : on le réduit au strict minimum
        config.width = 2
        config.height = 2
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1)
        let filter = SCContentFilter(display: display, excludingApplications: [], exceptingWindows: [])
        let stream = SCStream(filter: filter, configuration: config, delegate: self)
        try stream.addStreamOutput(self, type: .audio, sampleHandlerQueue: DispatchQueue(label: "audio"))
        try stream.addStreamOutput(self, type: .screen, sampleHandlerQueue: DispatchQueue(label: "screen"))
        try await stream.startCapture()
        self.stream = stream
        log("ready")
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio, sampleBuffer.isValid,
              let format = sampleBuffer.formatDescription?.audioStreamBasicDescription else { return }
        if !announced {
            announced = true
            log("format \(Int(format.mSampleRate)) Hz, \(format.mChannelsPerFrame) canal(aux), flags \(format.mFormatFlags)")
        }
        // float32 non entrelacé : un tampon par canal ; on garde le premier (mono demandé)
        try? sampleBuffer.withAudioBufferList { buffers, _ in
            guard let first = buffers.first, let data = first.mData else { return }
            out.write(Data(bytes: data, count: Int(first.mDataByteSize)))
        }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        log("stopped: \(error.localizedDescription)")
        exit(2)
    }
}

let capture = Capture()
Task {
    do {
        try await capture.start()
    } catch {
        log("error: \(error.localizedDescription)")  // le plus souvent : autorisation refusée
        exit(1)
    }
}
// MacDictée ferme notre stdin pour arrêter la capture
DispatchQueue.global().async {
    _ = FileHandle.standardInput.readDataToEndOfFile()
    exit(0)
}
dispatchMain()
